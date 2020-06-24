
HIPCC = /opt/rocm/bin/hipcc
GCC   = g++

INC  += -I/opt/rocm/include
LIB   = -L/opt/rocm/lib
LIB  += -lhip_hcc

EXE = a.out

all: lib test

lib:
	$(HIPCC) -O3 -fno-gpu-rdc -fPIC -c ast_gemm.cpp
	$(GCC) -shared -fPIC -o libastaux.so            ast_gemm.o

test:
	$(GCC) -D__HIP_PLATFORM_HCC__ -c main.cpp  $(INC)
	$(GCC) -fopenmp -o $(EXE) main.o ./libastaux.so  $(LIB)

clean:
	rm -f *.o *.so $(EXE)
