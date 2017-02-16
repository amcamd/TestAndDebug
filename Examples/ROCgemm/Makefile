#rocBLAS is built in the directory $(HOME)/repos/rocBLAS-develop-build
ROCBLASBUILD=buildTensile
ROCBLASPACKAGE=buildTensile
HIP_INCLUDE=/opt/rocm/hip/include
ROCBLAS_INCLUDE=$(HOME)/repos/$(ROCBLASPACKAGE)/library-package/include
ROCBLAS_LIB_PATH=$(HOME)/repos/$(ROCBLASBUILD)/library-build/src
ROCBLAS_LIB=rocblas-hcc-d 
TENSILE_LIB=rocblas-tensile
LDFLAGS=-lm -L$(ROCBLAS_LIB_PATH) -l$(ROCBLAS_LIB) -l$(TENSILE_LIB)
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
