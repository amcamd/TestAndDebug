#ROCBLAS_INSTALL_DIR=/opt/rocm/rocblas
ROCBLAS_INSTALL_DIR=/home/achapman/repos/amcamd/ldc0/rocBLAS/build/release/rocblas-install
ROCBLAS_INCLUDE=$(ROCBLAS_INSTALL_DIR)/include
ROCBLAS_LIB_PATH=$(ROCBLAS_INSTALL_DIR)/lib
ROCBLAS_LIB=rocblas
HIP_INCLUDE=/opt/rocm/hip/include
LDFLAGS=-L$(ROCBLAS_LIB_PATH) -l$(ROCBLAS_LIB)
LD=hipcc
CFLAGS=-I$(ROCBLAS_INCLUDE) -I$(HIP_INCLUDE) -msse3 -march=haswell
CPP=hipcc
OBJ=ldc0.o
EXE=ldc0

%.o: %.cpp
		$(CPP) -c -o $@ $< $(CFLAGS)

$(EXE) : $(OBJ)
		$(LD) $(OBJ) $(LDFLAGS) -o $@ 

clean:
		rm -f $(EXE) $(OBJ)
