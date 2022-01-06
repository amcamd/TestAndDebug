
mkfile_path := $(abspath $(lastword $(MAKEFILE_LIST)))
#current_dir := $(notdir $(patsubst %/,%,$(dir $(mkfile_path))))
mkfile_dir := $(dir $(mkfile_path))

# directory where rocBLAS is installed
ROCBLAS_INSTALL_DIR=/home/achapman/repos/ROCmSoftwarePlatform/rocBLAS/build/release/rocblas-install/rocblas
ROCBLAS_INCLUDE=$(ROCBLAS_INSTALL_DIR)/include
ROCBLAS_LIB_PATH=$(ROCBLAS_INSTALL_DIR)/lib
ROCBLAS_LIB=rocblas

HIPCC = /opt/rocm/bin/hipcc
GCC = g++

INC  += -I/opt/rocm/include -I$(ROCBLAS_INCLUDE)
LIB = libsymm.so
LIB_RPATH = -Wl,-rpath=$(mkfile_dir) -Wl,-rpath=$(ROCBLAS_LIB_PATH)

EXE = a.out
SRC = symm_reference.cpp symm_block_recursive.cpp
OBJ = $(subst .cpp,.o,$(SRC))

all: lib test

lib:
	$(HIPCC) -O3 -fno-gpu-rdc -fPIC $(INC) -c $(SRC)
	$(GCC) -shared -fPIC -o $(LIB) $(OBJ)

test:
	$(GCC) -D__HIP_PLATFORM_HCC__ -c main.cpp  $(INC)
	$(HIPCC) -o $(EXE) main.o $(INC) $(LIB_RPATH) -L$(ROCBLAS_LIB_PATH) $(LIB) -l$(ROCBLAS_LIB) 

clean:
	rm -f *.o $(LIB) $(EXE)
