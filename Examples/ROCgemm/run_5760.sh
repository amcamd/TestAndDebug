#!/bin/sh

rocm-smi -d 0 --setsclk 7
rocm-smi -d 0 --setfan 255
sleep 2
rocm-smi -d 0 -g

./rocblas_sgemm_example -m5760 -n5760 -k5760 -a5760 -b5760 -c5760 -tNT -ov -fy
rocm-smi -c | grep "GPU Clock Level" 
rocm-smi -d 0 --setsclk 7 > /dev/null

./rocblas_sgemm_example -m5760 -n5760 -k5760 -a5760 -b5760 -c5760 -tNN -ov -fn
rocm-smi -c | grep "GPU Clock Level" 
rocm-smi -d 0 --setsclk 7 > /dev/null

./rocblas_sgemm_example -m5760 -n5760 -k5760 -a5760 -b5760 -c5760 -tTT -ov -fn
rocm-smi -c | grep "GPU Clock Level" 
rocm-smi -d 0 --setsclk 7 > /dev/null

./rocblas_sgemm_example -m5760 -n5760 -k5760 -a5760 -b5760 -c5760 -tTN -ov -fn
rocm-smi -c | grep "GPU Clock Level" 
rocm-smi -d 0 --setsclk 7 > /dev/null

rocm-smi -d 0 -g
rocm-smi -d 0 --setfan 50

