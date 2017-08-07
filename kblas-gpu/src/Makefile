#Assumes rocBLAS is installed in the directory ROCBLAS_INSTALL_DIR

KBLAS_INCLUDE=../include

LD=hipcc
CFLAGS=-I$(KBLAS_INCLUDE)
CPP=hipcc
OBJ=ssymv.o

$(OBJ) :

%.o: %.cpp
	$(CPP) -c -o $@ $< $(CFLAGS)

clean:
	rm -f $(OBJ)
