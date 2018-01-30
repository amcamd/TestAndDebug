#!/bin/sh

rocm-smi -d 0 --setsclk 6
sleep 2
rocm-smi -d 0 -g
./rocblas_sgemm_example -m512 -n512 -k512 -a512 -b512 -c512 -tNT -ov -fy
rocm-smi -c | grep "GPU Clock Level" 
rocm-smi -d 0 --setsclk 6 > /dev/null
./rocblas_sgemm_example -m1024 -n1024 -k1024 -a1024 -b1024 -c1024 -tNT -ov -fn
rocm-smi -c | grep "GPU Clock Level" 
rocm-smi -d 0 --setsclk 6 > /dev/null
./rocblas_sgemm_example -m1536 -n1536 -k1536 -a1536 -b1536 -c1536 -tNT -ov -fn
rocm-smi -c | grep "GPU Clock Level" 
rocm-smi -d 0 --setsclk 6 > /dev/null
./rocblas_sgemm_example -m2048 -n2048 -k2048 -a2048 -b2048 -c2048 -tNT -ov -fn
rocm-smi -c | grep "GPU Clock Level" 
rocm-smi -d 0 --setsclk 6 > /dev/null
./rocblas_sgemm_example -m2560 -n2560 -k2560 -a2560 -b2560 -c2560 -tNT -ov -fn
rocm-smi -c | grep "GPU Clock Level" 
rocm-smi -d 0 --setsclk 6 > /dev/null
./rocblas_sgemm_example -m3072 -n3072 -k3072 -a3072 -b3072 -c3072 -tNT -ov -fn
rocm-smi -c | grep "GPU Clock Level" 
rocm-smi -d 0 --setsclk 6 > /dev/null
./rocblas_sgemm_example -m3584 -n3584 -k3584 -a3584 -b3584 -c3584 -tNT -ov -fn
rocm-smi -c | grep "GPU Clock Level" 
rocm-smi -d 0 --setsclk 6 > /dev/null
./rocblas_sgemm_example -m4096 -n4096 -k4096 -a4096 -b4096 -c4096 -tNT -ov -fn
rocm-smi -c | grep "GPU Clock Level" 
rocm-smi -d 0 --setsclk 6 > /dev/null
./rocblas_sgemm_example -m4608 -n4608 -k4608 -a4608 -b4608 -c4608 -tNT -ov -fn
rocm-smi -c | grep "GPU Clock Level" 
rocm-smi -d 0 --setsclk 6 > /dev/null
./rocblas_sgemm_example -m5120 -n5120 -k5120 -a5120 -b5120 -c5120 -tNT -ov -fn
rocm-smi -c | grep "GPU Clock Level" 
rocm-smi -d 0 --setsclk 6 > /dev/null
./rocblas_sgemm_example -m5632 -n5632 -k5632 -a5632 -b5632 -c5632 -tNT -ov -fn
rocm-smi -c | grep "GPU Clock Level" 
rocm-smi -d 0 --setsclk 6 > /dev/null
./rocblas_sgemm_example -m6144 -n6144 -k6144 -a6144 -b6144 -c6144 -tNT -ov -fn
rocm-smi -c | grep "GPU Clock Level" 
rocm-smi -d 0 --setsclk 6 > /dev/null
./rocblas_sgemm_example -m6656 -n6656 -k6656 -a6656 -b6656 -c6656 -tNT -ov -fn
rocm-smi -c | grep "GPU Clock Level" 
rocm-smi -d 0 --setsclk 6 > /dev/null
./rocblas_sgemm_example -m7168 -n7168 -k7168 -a7168 -b7168 -c7168 -tNT -ov -fn
rocm-smi -c | grep "GPU Clock Level" 
rocm-smi -d 0 --setsclk 6 > /dev/null
./rocblas_sgemm_example -m7680 -n7680 -k7680 -a7680 -b7680 -c7680 -tNT -ov -fn
rocm-smi -c | grep "GPU Clock Level" 
rocm-smi -d 0 --setsclk 6 > /dev/null
./rocblas_sgemm_example -m8192 -n8192 -k8192 -a8192 -b8192 -c8192 -tNT -ov -fn
rocm-smi -c | grep "GPU Clock Level" 
rocm-smi -d 0 --setsclk 6 > /dev/null
./rocblas_sgemm_example -m8704 -n8704 -k8704 -a8704 -b8704 -c8704 -tNT -ov -fn
rocm-smi -c | grep "GPU Clock Level" 
rocm-smi -d 0 --setsclk 6 > /dev/null
./rocblas_sgemm_example -m9216 -n9216 -k9216 -a9216 -b9216 -c9216 -tNT -ov -fn
rocm-smi -c | grep "GPU Clock Level" 

rocm-smi -d 0 -g
