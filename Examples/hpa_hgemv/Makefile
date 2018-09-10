
#ROCBLAS_INSTALL_DIR=$(HOME)/repos/rocBLAS/build/library-package
ROCBLAS_INSTALL_DIR=/opt/rocm/rocblas

ROCBLAS_INCLUDE=$(ROCBLAS_INSTALL_DIR)/include
ROCBLAS_LIB_PATH=$(ROCBLAS_INSTALL_DIR)/lib

ROCBLAS_LIB=rocblas

LDFLAGS=-L$(ROCBLAS_LIB_PATH) -l$(ROCBLAS_LIB) -fPIC -mcpu=gfx906
LD=hipcc
CFLAGS=-I$(ROCBLAS_INCLUDE) -fPIC -mcpu=gfx906
CPP=hipcc
OBJ=hpa_hgemv.o
EXE=hpa_hgemv

%.o: %.cpp
	$(CPP) -c -o $@ $< $(CFLAGS)

$(EXE) : $(OBJ)
	$(LD) $(LDFLAGS) $(OBJ) -o $@ 

clean:
	rm -f $(EXE) $(OBJ)
