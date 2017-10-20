#!/bin/sh

rocm-smi -d 0 --setsclk 3
sleep 2
rocm-smi -d 0 -g

./rocblas_sgemm_example -m128 -n128 -k128 -a128 -b128 -c128 -tNT -ot -fy
./rocblas_sgemm_example -m256 -n256 -k256 -a256 -b256 -c256 -tNT -ot -fn
./rocblas_sgemm_example -m384 -n384 -k384 -a384 -b384 -c384 -tNT -ot -fn
./rocblas_sgemm_example -m512 -n512 -k512 -a512 -b512 -c512 -tNT -ot -fn
./rocblas_sgemm_example -m640 -n640 -k640 -a640 -b640 -c640 -tNT -ot -fn
./rocblas_sgemm_example -m768 -n768 -k768 -a768 -b768 -c768 -tNT -ot -fn
./rocblas_sgemm_example -m896 -n896 -k896 -a896 -b896 -c896 -tNT -ot -fn
./rocblas_sgemm_example -m1024 -n1024 -k1024 -a1024 -b1024 -c1024 -tNT -ot -fn
#./rocblas_sgemm_example -m2048 -n2048 -k2048 -a2048 -b2048 -c2048 -tNT -ov -fn
#./rocblas_sgemm_example -m2176 -n2176 -k2176 -a2176 -b2176 -c2176 -tNT -ov -fn
#./rocblas_sgemm_example -m4096 -n4096 -k4096 -a4096 -b4096 -c4096 -tNT -ov -fn
#./rocblas_sgemm_example -m4992 -n4992 -k4992 -a4992 -b4992 -c4992 -tNT -ov -fn
#./rocblas_sgemm_example -m5120 -n5120 -k5120 -a5120 -b5120 -c5120 -tNT -ov -fn
#./rocblas_sgemm_example -m-n-k-a-b-c-tNT

rocm-smi -d 0 -g
