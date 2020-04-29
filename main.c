//
// a code to demonstrate the use of ParMETIS
//
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <mpi.h>
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
// Function to write the adjacency structure to files by unrolling the arrays
//
int write_adjacency( MPI_Comm comm,
                     idx_t* idst, idx_t* iadj, idx_t* jadj,
                     idx_t* vwgt, idx_t* ewgt,
                     char string[] )
{
   int irank,nrank;
   idx_t i,j;
   idx_t nv;
   FILE *fp;
   char fname[256];


   MPI_Comm_rank( comm, &irank );
   MPI_Comm_size( comm, &nrank );

   nv = idst[nrank];
   if( irank == 0 ) {
      fprintf( stdout, "Number of vertices: %ld \n",(long) nv );
      fprintf( stdout, "Writing adjacency to \"%s_*.dat\"\n", string );
   }

   sprintf( fname, "%s_%.6d.dat", string, irank );
   fp = fopen( fname, "w" );
   fprintf( fp, " %ld \n", (long) (idst[irank+1] - idst[irank]) );
   for( i=0; i <= idst[irank+1] - idst[irank]; ++i ) {
      fprintf( fp, "  %ld \n", (long) iadj[i] );
   }
   for( i=0; i < idst[irank+1] - idst[irank] ;++i ) {
      for(j=iadj[i];j<iadj[i+1];++j) {
         fprintf( fp, "   %ld \n", (long) jadj[j] );
      }
   }

   if( vwgt != NULL ) {
      for( i=0; i < idst[irank+1] - idst[irank]; ++i ) {
         fprintf( fp, "   %ld \n", (long) vwgt[i] );
      }
   }

   if( ewgt != NULL ) {
      for( i=0; i < idst[irank+1] - idst[irank] ;++i ) {
         for(j=iadj[i];j<iadj[i+1];++j) {
            fprintf( fp, "    %ld \n", (long) ewgt[j] );
         }
      }
   }

   fclose( fp );

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
// Function to create an adjacency structure for a 2D Laplacian
//
int make_adjacency_Laplacian2D( MPI_Comm comm, idx_t nx, idx_t ny,
                     idx_t** idst, idx_t** iadj, idx_t** jadj,
                     idx_t** vwgt, idx_t** ewgt )
{
   idx_t nv,nlv,irem;
   idx_t ii,jj;
   idx_t i,j;
   int irank,nrank,n;
   size_t isize;
   idx_t *_idst, *_iadj, *_jadj, *_vwgt, *_ewgt;


   MPI_Comm_rank( comm, &irank );
   MPI_Comm_size( comm, &nrank );

   // (almost) equally distribute vertices
   nv = nx*ny;
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
   isize = (size_t) (5*nlv);    // over-allocated at boundaries (safe)
   _jadj = (idx_t *) malloc( isize * sizeof(idx_t) );
   isize = (size_t) nlv;
   _vwgt = (idx_t *) malloc( isize * sizeof(idx_t) );
   isize = (size_t) (5*nlv);    // over-allocated at boundaries (safe)
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
   for(jj=0;jj<ny;++jj) {
   for(ii=0;ii<nx;++ii) {

      i = jj*nx + ii;              // global index of vertex

      if( _idst[irank] <= i && i < _idst[irank+1] ) {
         i -= _idst[irank];        // shift to local distribution index

         _iadj[i+1] = _iadj[i];   // end of vertex adjacency (row)
         _vwgt[i] = 1;            // vertex weight

         // self adjacency
         _jadj[ _iadj[i+1] ] = _idst[irank] + i;
         _ewgt[ _iadj[i+1] ] = 1;   // edge weight
         _iadj[i+1] += 1;

         // four other vertices
         if( 0 < ii )  {
            _jadj[ _iadj[i+1] ] = jj*nx + ii - 1;
            _ewgt[ _iadj[i+1] ] = 1;   // edge weight
            _iadj[i+1] += 1;
         }

         if( ii < nx-1 ) {
            _jadj[ _iadj[i+1] ] = jj*nx + ii + 1;
            _ewgt[ _iadj[i+1] ] = 1;   // edge weight
            _iadj[i+1] += 1;
         }

         if( 0 < jj )  {
            _jadj[ _iadj[i+1] ] = (jj-1)*nx + ii;
            _ewgt[ _iadj[i+1] ] = 1;   // edge weight
            _iadj[i+1] += 1;
         }

         if( jj < ny-1 ) {
            _jadj[ _iadj[i+1] ] = (jj+1)*nx + ii;
            _ewgt[ _iadj[i+1] ] = 1;   // edge weight
            _iadj[i+1] += 1;
         }

      }  // condition of locality of vertex in the distribution

   }}    // sweep over 2D domain vertices
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
// Function to partition with ParMETIS
//
int partition( MPI_Comm comm,
               idx_t* idst, idx_t* iadj, idx_t* jadj,
               idx_t* vwgt, idx_t* ewgt,
               idx_t npart,
               idx_t** ipv )
{
   int irank,nrank,n,ierr=0;
   size_t isize;
   idx_t ncon = 1;
   idx_t inum = 0;
   idx_t iflag = 0;
   real_t *r1, ubvec[1] = { 1.05 };
   idx_t nedge, *_ipv;
   idx_t iopt[4];
   double t1;


   MPI_Comm_rank( comm, &irank );
   MPI_Comm_size( comm, &nrank );

   isize = (size_t) (idst[irank+1] - idst[irank]);
   _ipv = (idx_t *) malloc( isize * sizeof(idx_t) );
   isize = (size_t) nrank;
   r1 = (real_t *) malloc( isize * sizeof(idx_t) );
   if( _ipv == NULL || r1 == NULL ) ierr = 1;
   MPI_Allreduce( MPI_IN_PLACE, &ierr, 1, MPI_INT, MPI_SUM, comm );
   if( ierr != 0 ) {
      if( irank == 0 )
         fprintf( stdout, "Could not allocate partition vector\n");
      if( _ipv != NULL ) free( _ipv );
      if( r1 != NULL ) free( r1 );
      return -1;
   }

   for(n=0;n<npart;++n) r1[n] = 1.0/((real_t) npart);

   if( vwgt != NULL ) iflag += 1;
   if( ewgt != NULL ) iflag += 2;
   if( irank == 0 ) fprintf( stdout, "Weights flag: %d \n",(int) iflag);

   iopt[0] = 1;    // non-default options (0 = default)
   iopt[1] = 0;    // info returned
   iopt[1] = PARMETIS_DBGLVL_TIME | PARMETIS_DBGLVL_INFO |
             PARMETIS_DBGLVL_PROGRESS | PARMETIS_DBGLVL_REFINEINFO |
             PARMETIS_DBGLVL_MATCHINFO | PARMETIS_DBGLVL_RMOVEINFO |
             PARMETIS_DBGLVL_REMAP;
   iopt[1] = PARMETIS_DBGLVL_TIME | PARMETIS_DBGLVL_INFO ;
   iopt[2] = 0;    // random seed
   iopt[3] = PARMETIS_PSR_UNCOUPLED;    // # part != # processors

   t1 = MPI_Wtime();
   nedge = 0;
   ierr = ParMETIS_V3_PartKway(
             idst, iadj, jadj, vwgt, ewgt,
             &iflag, &inum, &ncon, &npart, r1, ubvec, iopt, &nedge,
             _ipv, &comm );
   t1 = MPI_Wtime() - t1;
   if( irank == 0 ) fprintf( stdout, "Partitioned in %lf sec \n",t1);

   *ipv = _ipv;

   free( r1 );

   return 0;
}


//
// Driver for the 1D Laplacian partitioing
//
int driver_test1( MPI_Comm comm, idx_t nv )
{
   int irank,nrank;
   idx_t *idst,*iadj,*jadj, *ipv;
   idx_t *vwgt=NULL,*ewgt=NULL;


   MPI_Comm_rank( comm, &irank );
   MPI_Comm_size( comm, &nrank );

   (void) make_adjacency_Laplacian1D( comm, nv, &idst, &iadj, &jadj,
                                      &vwgt, &ewgt );

   (void) check_adjacency( comm, idst, iadj, jadj );

#ifdef _DEBUG2_
   (void) dump_adjacency( comm, idst, iadj, jadj );
#endif

   (void) partition( comm, idst, iadj, jadj, vwgt, ewgt, (idx_t) nrank, &ipv );

   free( ipv );
   free( ewgt );
   free( vwgt );
   free( jadj );
   free( iadj );
   free( idst );

   return 0;
}


//
// Driver for the 2D Laplacian partitioing
//
int driver_test2( MPI_Comm comm, idx_t nx, idx_t ny )
{
   int irank,nrank;
   idx_t *idst,*iadj,*jadj, *ipv;
   idx_t *vwgt=NULL,*ewgt=NULL;


   MPI_Comm_rank( comm, &irank );
   MPI_Comm_size( comm, &nrank );

   (void) make_adjacency_Laplacian2D( comm, nx, ny, &idst, &iadj, &jadj,
                                      &vwgt, &ewgt );

   (void) check_adjacency( comm, idst, iadj, jadj );

#ifdef _DEBUG2_
   (void) dump_adjacency( comm, idst, iadj, jadj );
#endif

   (void) write_adjacency( comm, idst, iadj, jadj, vwgt, ewgt, "my_adj" );

   (void) partition( comm, idst, iadj, jadj, vwgt, ewgt, (idx_t) nrank, &ipv );

   free( ipv );
   free( ewgt );
   free( vwgt );
   free( jadj );
   free( iadj );
   free( idst );

   return 0;
}


//
// Driver for partitioning an adjacency structure that is read from files
//
int driver_test3( MPI_Comm comm, char string[] )
{
   int irank,nrank;
   idx_t i,j;
   idx_t *idst,*iadj,*jadj, *ipv;
   idx_t *vwgt=NULL,*ewgt=NULL;
   FILE *fp;
   char *fname;
   int ilen,n;
   long nlv,nv;


   MPI_Comm_rank( comm, &irank );
   MPI_Comm_size( comm, &nrank );

   ilen = 0;
   while( string[ilen] != '\0' ) ++ilen;
   ilen += 20;
   fname = (char *) malloc( (size_t) ilen );
   if( fname == NULL ) {
      fprintf( stdout, "Could not allocate filename string \n");
      return -1;
   }

   sprintf( fname, "%s_%.6d.dat", string, irank );
   if( irank == 0 )
      fprintf( stdout, "Filename: \"%s\"\n", fname );
   fp = fopen( fname, "r" );

   fgets( fname, ilen, fp );
   sscanf( fname, "%ld", &nlv );
   if( irank == 0 )
      fprintf( stdout, "Number of vertices for rank 0: %ld \n", nlv );
   MPI_Allreduce( &nlv, &nv, 1, MPI_LONG, MPI_SUM, comm );
   if( irank == 0 )
      fprintf( stdout, "Total number of vertices: %ld \n", nv );

   idst = (idx_t *) malloc( ((size_t) nrank+1) * sizeof(idx_t) );
   iadj = (idx_t *) malloc( ((size_t) nlv+1) * sizeof(idx_t) );
   vwgt = (idx_t *) malloc( ((size_t) nlv  ) * sizeof(idx_t) );

   for(n=0;n<nrank+1;++n) idst[n] = 0;
   idst[irank+1] = (idx_t) nlv;
   MPI_Allreduce( MPI_IN_PLACE, idst, nrank+1, MPI_INT, MPI_SUM, comm );
   for(n=0;n<nrank;++n) idst[n+1] += idst[n];
#ifdef _DEBUG_
   if( irank == 0 )
      fprintf( stdout, "Vertices breakdown: \n");
   for(n=0;n<nrank;++n) {
      if( irank == n )
         fprintf( stdout, " partition %d  size (vertices) %ld \n",n, (long)nlv);
      MPI_Barrier( comm );
   }
#endif

   for( i=0; i < idst[irank+1] - idst[irank] + 1 ; ++i ) {
      long int itmp;

      fgets( fname, ilen, fp );
      sscanf( fname, "%ld", &itmp );
      iadj[i] = (idx_t) itmp;
   }

   jadj = (idx_t *) malloc( ((size_t) (iadj[nlv])) * sizeof(idx_t) );
   ewgt = (idx_t *) malloc( ((size_t) (iadj[nlv])) * sizeof(idx_t) );

   for( j=0; j < iadj[ nlv ] ; ++j ) {
      long int itmp;

      fgets( fname, ilen, fp );
      sscanf( fname, "%ld", &itmp );
      jadj[j] = (idx_t) itmp;
   }

   for( i=0; i < idst[irank+1] - idst[irank] ; ++i ) {
      long int itmp;

      fgets( fname, ilen, fp );
      sscanf( fname, "%ld", &itmp );
      vwgt[i] = (idx_t) itmp;
   }

   for( j=0; j < iadj[ nlv ] ; ++j ) {
      long int itmp;

      fgets( fname, ilen, fp );
      sscanf( fname, "%ld", &itmp );
      ewgt[j] = (idx_t) itmp;
   }

#ifdef _DEBUG_
   (void) dump_adjacency( comm, idst, iadj, jadj );
#endif

   free( fname );

   (void) check_adjacency( comm, idst, iadj, jadj );

   (void) partition( comm, idst, iadj, jadj, vwgt, ewgt, (idx_t) nrank, &ipv );

   free( ipv );
   free( ewgt );
   free( vwgt );
   free( jadj );
   free( iadj );
   free( idst );

   return 0;
}

//
// Driver
//
int main( int argc, char *argv[] )
{
   MPI_Comm comm;
   int irank,nrank;
   idx_t nv, nx,ny;


   MPI_Init( &argc, &argv );
   comm = MPI_COMM_WORLD;
   MPI_Comm_rank( comm, &irank );
   MPI_Comm_size( comm, &nrank );

   nv = 350;
   (void) driver_test1( comm, nv );

   nx = 30;
   ny = 20;
   (void) driver_test2( comm, nx, ny );

   (void) driver_test3( comm, "my_adj" );

   MPI_Finalize();

   return EXIT_SUCCESS;
}

