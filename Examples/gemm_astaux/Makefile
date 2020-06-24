
HIPCC = /opt/rocm/bin/hipcc
GCC   = g++

INC  += -I/opt/rocm/include
LIB   = -L/opt/rocm/lib
LIB  += -lhip_hcc


all: lib test

lib:
	$(HIPCC) -O3 -fno-gpu-rdc -fPIC -c ast_gemm.cpp
	$(GCC) -shared -fPIC -o libastaux.so            ast_gemm.o

test:
	$(GCC) -D__HIP_PLATFORM_HCC__ -c test_gemm.cpp  $(INC)
	$(GCC) -fopenmp -o test_gemm  test_gemm.o ./libastaux.so  $(LIB)

clean:
	rm -f *.o *.so test_gemm
