# Makefile assumes rocBLAS is installed in /opt/rocm/rocblas

ROCBLAS_INSTALL_DIR=/opt/rocm/rocblas
ROCBLAS_INCLUDE=$(ROCBLAS_INSTALL_DIR)/include
ROCBLAS_LIB_PATH=$(ROCBLAS_INSTALL_DIR)/lib
ROCBLAS_LIB=rocblas
HIP_INCLUDE=/opt/rocm/hip/include
LDFLAGS=-L$(ROCBLAS_LIB_PATH) -l$(ROCBLAS_LIB)
LD=hipcc
CFLAGS=-I$(ROCBLAS_INCLUDE) -I$(HIP_INCLUDE)
CPP=hipcc
OBJ=rocblas_sscal_example.o
EXE=rocblas_sscal_example

%.o: %.cpp
	$(CPP) -c -o $@ $< $(CFLAGS)

$(EXE) : $(OBJ)
	$(LD) $(OBJ) $(LDFLAGS) -o $@ 

clean:
	rm -f $(EXE) $(OBJ)
