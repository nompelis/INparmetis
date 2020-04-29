###### Pre-processor directives to increase verbosity
DEBUG = -D   _DEBUG_

###### additional compile-time configuration
SPECIAL_OPTIONS = -D  _LINUX_

###### the ParMETIS library compile/link-time options
# (make sure you have a module that sets the variable METIS_HOME)
 INCLUDE = -I $(METIS_HOME)/include
 LIBS = -L $(METIS_HOME)/lib -lparmetis

###### C++ 
MPICXX = mpicxx
CXXOPTS = -Wall -fPIC -O0 -std=c++98 -pedantic
CXXOPTS = -Wall -fPIC -O0
CXXOPTS += $(SPECIAL_OPTIONS)

###### C
MPICC = mpicc
COPTS = -Wall -fPIC -O0
COPTS += $(SPECIAL_OPTIONS)

###### linker
LD = ld


###### targets ######

all:
	$(MPICC) $(DEBUG) $(COPTS) $(INCLUDE) main.c $(LIBS)

clean:
	rm -f *.o *.a a.out *.so *.a

