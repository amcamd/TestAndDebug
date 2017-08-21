#!/bin/bash

#set device 1 clock to level 7: 1000Mhz
#rocm-smi -d 1 --setsclk 7
#sleep 2
#rocm-smi -d 1 -g

./rocblas_sgemm_example > out00
./rocblas_sgemm_example > out01

#reset device 1 clock to default values
#rocm-smi -d 1 -r

grep "min,ave,max,rsd_gflops" out?? > out.summary
