
ROCBLAS_INSTALL_DIR=/home/achapman/repos/ROCmSoftwarePlatform/rocBLAS-internal/build_tensile/release/rocblas-install
ROCBLAS_INCLUDE=$(ROCBLAS_INSTALL_DIR)/include/rocblas
ROCBLAS_LIB_PATH=$(ROCBLAS_INSTALL_DIR)/lib
ROCBLAS_LIB=rocblas


HIPCC = /opt/rocm/bin/hipcc

INC  += -I/opt/rocm/include -I$(ROCBLAS_INCLUDE)
LIB   = -L/opt/rocm/lib -L$(ROCBLAS_LIB_PATH)
LIB  += -lhip_hcc -l$(ROCBLAS_LIB)

EXE = a.out

#all: lib test
all:     test

#lib:
#    $(HIPCC) -O3 -fno-gpu-rdc -fPIC $(INC) -c ast_gemm.cpp
#    $(GCC) -shared -fPIC -o libastaux.so            ast_gemm.o

test:
	$(HIPCC) -o $(EXE) main.cpp $(INC)
#   $(GCC) -D__HIP_PLATFORM_HCC__ -c main.cpp  $(INC)

clean:
	rm -f *.o *.so $(EXE)

