# Makefile assumes rocBLAS is built in directory $(HOME)/repos/rocBLAS-build
HIP_INCLUDE=/opt/rocm/hip/include
ROCBLAS_INCLUDE=$(HOME)/repos/rocBLAS-build/library-package/include
ROCBLAS_LIB_PATH=$(HOME)/repos/rocBLAS-build/library-package/lib
ROCBLAS_LIB=rocblas-hcc
TENSILE_LIB=tensile-rocblas
LDFLAGS=-lm -L$(ROCBLAS_LIB_PATH) -l$(ROCBLAS_LIB) -l$(TENSILE_LIB)
LD=hipcc
CFLAGS=-I$(ROCBLAS_INCLUDE) -I$(HIP_INCLUDE)
CPP=hipcc
OBJ=rocblas_sscal_example.o
EXE=rocblas_sscal_example

%.o: %.cpp
	$(CPP) -c -o $@ $< $(CFLAGS)

$(EXE) : $(OBJ)
	$(LD) $(LDFLAGS) $(OBJ) -o $@ 

clean:
	rm -f $(EXE) $(OBJ)
