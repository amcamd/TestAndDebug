#Assumes rocBLAS is installed in the directory ROCBLAS_INSTALL_DIR

HIP_INCLUDE=/opt/rocm/hip/include

#ROCBLAS_INSTALL_DIR=$(HOME)/repos/rocBLAS/build/library-package
ROCBLAS_INSTALL_DIR=/opt/rocm/rocblas

ROCBLAS_INCLUDE=$(ROCBLAS_INSTALL_DIR)/include
ROCBLAS_LIB_PATH=$(ROCBLAS_INSTALL_DIR)/lib

ROCBLAS_LIB=rocblas

LDFLAGS=-lm -L$(ROCBLAS_LIB_PATH) -l$(ROCBLAS_LIB)
LD=hipcc
CFLAGS=-I$(ROCBLAS_INCLUDE) -I$(HIP_INCLUDE)
CPP=hipcc
OBJ=rocblas_sgemm_example.o
EXE=rocblas_sgemm_example

%.o: %.cpp
	$(CPP) -c -o $@ $< $(CFLAGS)

$(EXE) : $(OBJ)
	$(LD) $(LDFLAGS) $(OBJ) -o $@ 

clean:
	rm -f $(EXE) $(OBJ)
