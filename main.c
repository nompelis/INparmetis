//
// a code to demonstrate the use of ParMETIS
//
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <parmetis.h>


//
// Function to dump the adjacency structure to files
//
int dump_adjacency( MPI_Comm comm,
                    idx_t* idst, idx_t* iadj, idx_t* jadj )
{
   int irank,nrank;
   idx_t i,j;
   idx_t nv;
   FILE *fp;
   char fname[32];


   MPI_Comm_rank( comm, &irank );
   MPI_Comm_size( comm, &nrank );

   nv = idst[nrank];
   if( irank == 0 ) fprintf( stdout, "Number of vertices: %ld \n",(long) nv );

   if( irank == 0 ) fprintf( stdout, "Dumping adjacency to files \n");
   sprintf( fname, "adj_%.6d.dat", irank );
   fp = fopen( fname, "w" );
   fprintf( fp, "### Adjacency structure for rank: %d \n",irank );
   fprintf( fp, "### (plot it with GnuPlot: `plot \"adj_NNNN.dat\" w l') \n");
   for( i=0; i < idst[irank+1] - idst[irank] ;++i ) {
      for(j=iadj[i];j<iadj[i+1];++j) {
         fprintf( fp, "%ld %ld \n",    // matrix form
                  (long) jadj[j], (long) (nv - (idst[irank]+i)) );
      }
   }
   fclose( fp );

   return 0;
}



//
// Function to check whether an adjacency structure goes out of bounds
//
int check_adjacency( MPI_Comm comm,
                     idx_t* idst, idx_t* iadj, idx_t* jadj )
{
   int irank,nrank;
   idx_t i,j,k;
   idx_t nv;
   long int nerr = 0;


   MPI_Comm_rank( comm, &irank );
   MPI_Comm_size( comm, &nrank );

   if( idst == NULL || iadj == NULL || jadj == NULL ) {
      fprintf( stdout, "Rank %d was given a null adjaceency \n",irank);
      return 1;
   }

   nv = idst[nrank];
   if( irank == 0 ) fprintf( stdout, "Number of vertices: %ld \n",(long) nv );

   for( i=0; i < idst[irank+1] - idst[irank] ;++i ) {
      for(j=iadj[i];j<iadj[i+1];++j) {
         k = jadj[j];
         if( k < 0 ) ++nerr;
         if( k >= nv ) ++nerr;
      }
   }

   MPI_Allreduce( MPI_IN_PLACE, &nerr, 1, MPI_LONG, MPI_SUM, comm );
   if( nerr != 0 ) {
      if( irank == 0 ) fprintf( stdout, "Found out-of-bounds error \n");
      return 2;
   } else {
      if( irank == 0 ) fprintf( stdout, "No out-of-bounds problem \n");
   }

   return 0;
}


//
// Function to create an adjacency structure for a 1D Laplacian
//
int make_adjacency_Laplacian1D( MPI_Comm comm, idx_t nv,
                     idx_t** idst, idx_t** iadj, idx_t** jadj,
                     idx_t** vwgt, idx_t** ewgt )
{
   idx_t nlv,irem;
   idx_t i,j;
   int irank,nrank,n;
   size_t isize;
   idx_t *_idst, *_iadj, *_jadj, *_vwgt, *_ewgt;


   MPI_Comm_rank( comm, &irank );
   MPI_Comm_size( comm, &nrank );

   // (almost) equally distribute vertices
   nlv = nv / ((idx_t) nrank);
   irem = nv - nlv*((idx_t) nrank);
   if( ((idx_t) irank) < irem ) ++nlv;
#ifdef _DEBUG_
   if( irank == 0 )
      fprintf( stdout, "Vertices breakdown: \n");
   for(n=0;n<nrank;++n) {
      if( irank == n )
         fprintf( stdout, " partition %d  size (vertices) %ld \n",n, (long)nlv);
      MPI_Barrier( comm );
   }
#endif

   // allocate emory for adjacency structure
   isize = (size_t) (nrank+1);
   _idst = (idx_t *) malloc( isize * sizeof(idx_t) );
   isize = (size_t) (nlv+1);
   _iadj = (idx_t *) malloc( isize * sizeof(idx_t) );
   isize = (size_t) (3*nlv);    // over-allocated at boundaries (safe)
   _jadj = (idx_t *) malloc( isize * sizeof(idx_t) );
   isize = (size_t) nlv;
   _vwgt = (idx_t *) malloc( isize * sizeof(idx_t) );
   isize = (size_t) (3*nlv);    // over-allocated at boundaries (safe)
   _ewgt = (idx_t *) malloc( isize * sizeof(idx_t) );
   n = 0;
   if( _idst == NULL || _iadj == NULL || _jadj == NULL ||
       _vwgt == NULL || _ewgt == NULL ) n = 1;
   MPI_Allreduce( MPI_IN_PLACE, &n, 1, MPI_INT, MPI_SUM, comm );
   if( n != 0 ) {
      if( irank == 0 ) fprintf( stdout, "Memory allocation error\n");
      if( _idst != NULL ) free( _idst );
      if( _iadj != NULL ) free( _iadj );
      if( _jadj != NULL ) free( _jadj );
      if( _vwgt != NULL ) free( _vwgt );
      if( _ewgt != NULL ) free( _ewgt );
      return -1;
#ifdef _DEBUG_
   } else {
      if( irank == 0 )
         fprintf( stdout, "Adjacency structure for 1D laplactian allocated\n");
#endif
   }

   // form distribution array
   for(n=0;n<nrank+1;++n) _idst[n] = 0;
   _idst[irank+1] = nlv;
   MPI_Allreduce( MPI_IN_PLACE, _idst, nrank+1, MPI_INT, MPI_SUM, comm );
   for(n=0;n<nrank;++n) _idst[n+1] += _idst[n];
#ifdef _DEBUG_
   if( irank == 0 ) {
      fprintf( stdout, "Vertices distribution across ranks: \n");
      for(n=0;n<nrank;++n) {
         fprintf( stdout, " rank %6d  offset %ld \n",n,(long) _idst[n] );
      }
      fprintf( stdout, " end  ______  offset %ld \n",(long) _idst[nrank] );
   }
   MPI_Barrier( comm );
#endif

   // form adjacency structure
   _iadj[0] = 0;
   for(i=0;i<nlv;++i) {
      _iadj[i+1] = _iadj[i];   // end of vertex adjacency (row)
      _vwgt[i] = 1;            // vertex weight

      // left vertex
      if( _idst[irank] + i - 1 >= 0 ) {
         _jadj[ _iadj[i+1] ] = _idst[irank] + i - 1;
         _ewgt[ _iadj[i+1] ] = 1;   // edge weight
         _iadj[i+1] += 1;
      }

      // self adjacency
      _jadj[ _iadj[i+1] ] = _idst[irank] + i;
      _ewgt[ _iadj[i+1] ] = 1;   // edge weight
      _iadj[i+1] += 1;

      // right vertex
      if( _idst[irank] + i + 1 < nv ) {
         _jadj[ _iadj[i+1] ] = _idst[irank] + i + 1;
         _ewgt[ _iadj[i+1] ] = 1;   // edge weight
         _iadj[i+1] += 1;
      }
   }
#ifdef _DEBUG2_
{  FILE *fp;
   char fname[32];
   if( irank == 0 ) fprintf( stdout, "Dumping adjacency to files \n");
   sprintf( fname, "adj_%.6d.dat", irank );
   fp = fopen( fname, "w" );
   fprintf( fp, "### Adjacency structure for rank: %d \n",irank );
   fprintf( fp, "### (plot it with GnuPlot: `plot \"adj_NNNN.dat\" w l') \n");
   for(i=0;i<nlv;++i) {
      for(j=_iadj[i];j<_iadj[i+1];++j) {
         fprintf( fp, "%ld %ld \n",    // matrix form
                  (long) _jadj[j], (long) (nv - (_idst[irank]+i)) );
      }
   }
   fclose( fp );
}
#endif

   // return pointers
   *idst = _idst;
   *iadj = _iadj;
   *jadj = _jadj;
   if( vwgt != NULL ) {
      *vwgt = _vwgt;
   } else {
      free( _vwgt );
   }
   if( ewgt != NULL ) {
      *ewgt = _ewgt;
   } else {
      free( _ewgt );
   }

   return 0;
}


//
// Driver
//
int main( int argc, char *argv[] )
{
   MPI_Comm comm;
   int irank,nrank;
   idx_t nv,np;
   idx_t *idst,*iadj,*jadj;


   MPI_Init( &argc, &argv );
   comm = MPI_COMM_WORLD;
   MPI_Comm_rank( comm, &irank );
   MPI_Comm_size( comm, &nrank );


   nv = 35;
   (void) make_adjacency_Laplacian1D( comm, nv, &idst, &iadj, &jadj, NULL,NULL);

   (void) check_adjacency( comm, idst, iadj, jadj );

   (void) dump_adjacency( comm, idst, iadj, jadj );


   MPI_Finalize();

   return EXIT_SUCCESS;
}

