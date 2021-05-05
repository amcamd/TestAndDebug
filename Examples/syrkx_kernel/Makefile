
ROCBLAS_INSTALL_DIR=/home/achapman/repos/ROCmSoftwarePlatform/rocBLAS/build/release/rocblas-install/rocblas
ROCBLAS_INCLUDE=$(ROCBLAS_INSTALL_DIR)/include
ROCBLAS_LIB_PATH=$(ROCBLAS_INSTALL_DIR)/lib
ROCBLAS_LIB=rocblas


HIPCC = /opt/rocm/bin/hipcc
GCC   = g++

INC  += -I/opt/rocm/include -I$(ROCBLAS_INCLUDE)
LIB   = -L/opt/rocm/lib -L$(ROCBLAS_LIB_PATH)
LIB  += -lhip_hcc -l$(ROCBLAS_LIB)

EXE = a.out

all: lib test
#all:     test

lib:
	$(HIPCC) -O3 -fno-gpu-rdc -fPIC $(INC) -c gemm_batched.cpp
	$(GCC) -shared -fPIC -o libastaux.so            gemm_batched.o

test:
	$(GCC) -D__HIP_PLATFORM_HCC__ -c main.cpp  $(INC)
	$(HIPCC) -fopenmp -o $(EXE) main.o ./libastaux.so
#	$(HIPCC) -o $(EXE) main.cpp $(INC)
#   $(GCC) -D__HIP_PLATFORM_HCC__ -c main.cpp  $(INC)

clean:
	rm -f *.o *.so $(EXE)

