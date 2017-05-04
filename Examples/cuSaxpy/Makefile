
a.out :
	/usr/local/cuda-8.0/bin/nvcc -ccbin g++ -I../../common/inc  -m64 -gencode arch=compute_35,code=sm_35 saxpy.orig.cu

clean :
	rm -f a.out saxpy.orig.o
