#!/bin/sh

rocm-smi -d 0 --setsclk 7
rocm-smi -d 0 --setfan 255
sleep 2
rocm-smi -d 0 -g

./rocblas_sgemm_example -m1024 -n1024 -k1024 -a1025 -b1025 -c1025 -tNT -ov -fy
rocm-smi -c | grep "GPU Clock Level" 
rocm-smi -d 0 --setsclk 7 > /dev/null

./rocblas_sgemm_example -m1024 -n1024 -k1024 -a1025 -b1025 -c1025 -tNN -ov -fn
rocm-smi -c | grep "GPU Clock Level" 
rocm-smi -d 0 --setsclk 7 > /dev/null

./rocblas_sgemm_example -m1024 -n1024 -k1024 -a1025 -b1025 -c1025 -tTT -ov -fn
rocm-smi -c | grep "GPU Clock Level" 
rocm-smi -d 0 --setsclk 7 > /dev/null

./rocblas_sgemm_example -m1024 -n1024 -k1024 -a1025 -b1025 -c1025 -tTN -ov -fn
rocm-smi -c | grep "GPU Clock Level" 
rocm-smi -d 0 --setsclk 7 > /dev/null

rocm-smi -d 0 -g
rocm-smi -d 0 --setfan 50

