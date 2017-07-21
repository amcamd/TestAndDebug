# Makefile assumes rocBLAS is installed in $(HOME)/repos/rocBLAS/build/library-package
ROCBLAS_INSTALL_DIR=/opt/rocm/rocblas
ROCBLAS_INCLUDE=$(ROCBLAS_INSTALL_DIR)/include
ROCBLAS_LIB_PATH=$(ROCBLAS_INSTALL_DIR)/lib
ROCBLAS_LIB=rocblas-hcc
HIP_INCLUDE=/opt/rocm/hip/include
#LDFLAGS=-lm -L$(ROCBLAS_LIB_PATH) -l$(ROCBLAS_LIB)
LDFLAGS=-lm
LD=hipcc
CFLAGS=-I$(ROCBLAS_INCLUDE) -I$(HIP_INCLUDE)
CPP=hipcc
OBJ=atomic_add.o
EXE=atomic_add_example

%.o: %.cpp
	$(CPP) -c -o $@ $< $(CFLAGS)

$(EXE) : $(OBJ)
	$(LD) $(LDFLAGS) $(OBJ) -o $@ 

clean:
	rm -f $(EXE) $(OBJ)
