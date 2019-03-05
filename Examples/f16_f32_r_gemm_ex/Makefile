#ROCBLAS_INSTALL_DIR=/opt/rocm/rocblas
ROCBLAS_INSTALL_DIR=/home/achapman/repos/amcamd/miopen_failure/v2.1.1/rocBLAS/build/release/rocblas-install/rocblas
ROCBLAS_INCLUDE=$(ROCBLAS_INSTALL_DIR)/include
ROCBLAS_LIB_PATH=$(ROCBLAS_INSTALL_DIR)/lib
ROCBLAS_LIB=rocblas
HIP_INCLUDE=/opt/rocm/hip/include
LDFLAGS=-L$(ROCBLAS_LIB_PATH) -l$(ROCBLAS_LIB)
LD=hipcc
CFLAGS=-I$(ROCBLAS_INCLUDE) -I$(HIP_INCLUDE) -msse3 -march=haswell
CPP=hipcc
OBJ=f16_f32_r_gemm_ex_example.o
EXE=f16_f32_r_gemm_ex_example

%.o: %.cpp
		$(CPP) -c -o $@ $< $(CFLAGS)

$(EXE) : $(OBJ)
		$(LD) $(OBJ) $(LDFLAGS) -o $@ 

clean:
		rm -f $(EXE) $(OBJ)
