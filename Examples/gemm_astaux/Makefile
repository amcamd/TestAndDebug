
HIPCC = /opt/rocm/bin/hipcc
GCC   = g++

INC  += -I/opt/rocm/include
LIB   = -L/opt/rocm/lib
LIB  += -lhip_hcc

#MKL_INC  = -I/opt/intel/mkl/include
#MKL_LIB  = -L/opt/intel/mkl/lib/intel64 -Wl,--no-as-needed
#MKL_LIB += -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl

all: lib test

lib:
#	$(HIPCC) -O3 -fno-gpu-rdc -fPIC -c ast_gemv.cpp
	$(HIPCC) -O3 -fno-gpu-rdc -fPIC -c ast_gemm.cpp
#	$(HIPCC) -O3 -fno-gpu-rdc -fPIC -c ast_getri_tiny.cpp
#	$(HIPCC) -O3 -fno-gpu-rdc -fPIC -c ast_getri_midsize.cpp
	$(GCC) -shared -fPIC -o libastaux.so            ast_gemm.o
#	$(GCC) -shared -fPIC -o libastaux.so ast_gemv.o ast_gemm.o ast_getri_midsize.o ast_getri_tiny.o

test:
	$(GCC) -D__HIP_PLATFORM_HCC__ -c test_gemm.cpp  $(INC)
	$(GCC) -fopenmp -o test_gemm  test_gemm.o ./libastaux.so  $(LIB)
#	$(GCC) -D__HIP_PLATFORM_HCC__ -c test_util.cpp  $(INC) $(MKL_INC)
#	$(GCC) -D__HIP_PLATFORM_HCC__ -c test_getri.cpp $(INC) $(MKL_INC)
#	$(GCC) -fopenmp -o test_getri test_getri.o test_util.o libastaux.so $(LIB) $(MKL_LIB)

clean:
	rm -f *.o *.so test_gemm
