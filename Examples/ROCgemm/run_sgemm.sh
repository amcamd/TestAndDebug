#!/bin/bash

set device 1 clock to level 7: 1000Mhz
rocm-smi -d 1 --setsclk 7
sleep 2
rocm-smi -d 1 -g

./rocblas_sgemm_example > out00
./rocblas_sgemm_example > out01
./rocblas_sgemm_example > out02
./rocblas_sgemm_example > out03
./rocblas_sgemm_example > out04
./rocblas_sgemm_example > out05
./rocblas_sgemm_example > out06
./rocblas_sgemm_example > out07
./rocblas_sgemm_example > out08
./rocblas_sgemm_example > out09
./rocblas_sgemm_example > out10
./rocblas_sgemm_example > out11
./rocblas_sgemm_example > out12
./rocblas_sgemm_example > out13
./rocblas_sgemm_example > out14
./rocblas_sgemm_example > out15
./rocblas_sgemm_example > out16
./rocblas_sgemm_example > out17
./rocblas_sgemm_example > out18
./rocblas_sgemm_example > out19
./rocblas_sgemm_example > out20
./rocblas_sgemm_example > out21
./rocblas_sgemm_example > out22
./rocblas_sgemm_example > out23
./rocblas_sgemm_example > out24
./rocblas_sgemm_example > out25
./rocblas_sgemm_example > out26
./rocblas_sgemm_example > out27
./rocblas_sgemm_example > out28
./rocblas_sgemm_example > out29
./rocblas_sgemm_example > out30
./rocblas_sgemm_example > out31
./rocblas_sgemm_example > out32
./rocblas_sgemm_example > out33
./rocblas_sgemm_example > out34
./rocblas_sgemm_example > out35
./rocblas_sgemm_example > out36
./rocblas_sgemm_example > out37
./rocblas_sgemm_example > out38
./rocblas_sgemm_example > out39
./rocblas_sgemm_example > out40
./rocblas_sgemm_example > out41
./rocblas_sgemm_example > out42
./rocblas_sgemm_example > out43
./rocblas_sgemm_example > out44
./rocblas_sgemm_example > out45
./rocblas_sgemm_example > out46
./rocblas_sgemm_example > out47
./rocblas_sgemm_example > out48
./rocblas_sgemm_example > out49

reset device 1 clock to default values
rocm-smi -d 1 -r

grep "min,ave,max,rsd_gflops" out?? > out.summary
