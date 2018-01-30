#!/bin/sh

rocm-smi -d 0 --setsclk 6
sleep 2
rocm-smi -d 0 -g
./rocblas_sgemm_example -m1024 -n1024 -k1024 -a1024 -b1024 -c1024 -tNT -ov -fy
rocm-smi -c | grep "GPU Clock Level" 
rocm-smi -d 0 --setsclk 6 > /dev/null
./rocblas_sgemm_example -m2048 -n2048 -k2048 -a2048 -b2048 -c2048 -tNT -ov -fn
rocm-smi -c | grep "GPU Clock Level" 
rocm-smi -d 0 --setsclk 6 > /dev/null
./rocblas_sgemm_example -m3072 -n3072 -k3072 -a3072 -b3072 -c3072 -tNT -ov -fn
rocm-smi -c | grep "GPU Clock Level" 
rocm-smi -d 0 --setsclk 6 > /dev/null
./rocblas_sgemm_example -m4096 -n4096 -k4096 -a4096 -b4096 -c4096 -tNT -ov -fn
rocm-smi -c | grep "GPU Clock Level" 
rocm-smi -d 0 --setsclk 6 > /dev/null
./rocblas_sgemm_example -m5120 -n5120 -k5120 -a5120 -b5120 -c5120 -tNT -ov -fn
rocm-smi -c | grep "GPU Clock Level" 
rocm-smi -d 0 --setsclk 6 > /dev/null
./rocblas_sgemm_example -m6144 -n6144 -k6144 -a6144 -b6144 -c6144 -tNT -ov -fn
rocm-smi -c | grep "GPU Clock Level" 
rocm-smi -d 0 --setsclk 6 > /dev/null
./rocblas_sgemm_example -m7168 -n7168 -k7168 -a7168 -b7168 -c7168 -tNT -ov -fn
rocm-smi -c | grep "GPU Clock Level" 
rocm-smi -d 0 --setsclk 6 > /dev/null
./rocblas_sgemm_example -m8192 -n8192 -k8192 -a8192 -b8192 -c8192 -tNT -ov -fn
rocm-smi -c | grep "GPU Clock Level" 
rocm-smi -d 0 --setsclk 6 > /dev/null
./rocblas_sgemm_example -m9216 -n9216 -k9216 -a9216 -b9216 -c9216 -tNT -ov -fn
rocm-smi -c | grep "GPU Clock Level" 

rocm-smi -d 0 -g
