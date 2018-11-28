#!/bin/sh

rocm-smi -d 0 --setsclk 7
rocm-smi -d 0 --setfan 255
sleep 2
rocm-smi -d 0 -g


./rocblas_sgemm_example -m1760 -n800 -k1760 -a1760 -b1760 -c1760 -tTN -ov -fy
rocm-smi -c | grep "GPU Clock Level" 
rocm-smi -d 0 --setsclk 7 > /dev/null

./rocblas_sgemm_example -m1760 -n1600 -k1760 -a1760 -b1760 -c1760 -tTN -ov -fn
rocm-smi -c | grep "GPU Clock Level" 
rocm-smi -d 0 --setsclk 7 > /dev/null


./rocblas_sgemm_example -m1760  -n3200  -k1760 -a1760   -b1760  -c1760   -tTN -ov -fn
rocm-smi -c | grep "GPU Clock Level"                                                      
rocm-smi -d 0 --setsclk 7 > /dev/null                                                     
./rocblas_sgemm_example -m1760  -n6400  -k1760 -a1760   -b1760  -c1760   -tTN -ov -fn
rocm-smi -c | grep "GPU Clock Level"                                                      
rocm-smi -d 0 --setsclk 7 > /dev/null                                                     
./rocblas_sgemm_example -m2048  -n800   -k2048 -a2048   -b2048  -c2048   -tTN -ov -fn
rocm-smi -c | grep "GPU Clock Level"                                                      
rocm-smi -d 0 --setsclk 7 > /dev/null                                                     
./rocblas_sgemm_example -m2048  -n1600  -k2048 -a2048   -b2048  -c2048   -tTN -ov -fn
rocm-smi -c | grep "GPU Clock Level"                                                      
rocm-smi -d 0 --setsclk 7 > /dev/null                                                     
./rocblas_sgemm_example -m2048  -n3200  -k2048 -a2048   -b2048  -c2048   -tTN -ov -fn
rocm-smi -c | grep "GPU Clock Level"                                                      
rocm-smi -d 0 --setsclk 7 > /dev/null                                                     
./rocblas_sgemm_example -m2048  -n6400  -k2048 -a2048   -b2048  -c2048   -tTN -ov -fn
rocm-smi -c | grep "GPU Clock Level"                                                      
rocm-smi -d 0 --setsclk 7 > /dev/null                                                     
./rocblas_sgemm_example -m2560  -n800   -k2560 -a2560   -b2560  -c2560   -tTN -ov -fn
rocm-smi -c | grep "GPU Clock Level"                                                      
rocm-smi -d 0 --setsclk 7 > /dev/null                                                     
./rocblas_sgemm_example -m2560  -n1600  -k2560 -a2560   -b2560  -c2560   -tTN -ov -fn
rocm-smi -c | grep "GPU Clock Level"                                                      
rocm-smi -d 0 --setsclk 7 > /dev/null                                                     
./rocblas_sgemm_example -m2560  -n3200  -k2560 -a2560   -b2560  -c2560   -tTN -ov -fn
rocm-smi -c | grep "GPU Clock Level"                                                      
rocm-smi -d 0 --setsclk 7 > /dev/null                                                     
./rocblas_sgemm_example -m2560  -n6400  -k2560 -a2560   -b2560  -c2560   -tTN -ov -fn
rocm-smi -c | grep "GPU Clock Level"                                                      
rocm-smi -d 0 --setsclk 7 > /dev/null                                                     
./rocblas_sgemm_example -m2048  -n400   -k512  -a512    -b512   -c2048   -tTN -ov -fn
rocm-smi -c | grep "GPU Clock Level"                                                      
rocm-smi -d 0 --setsclk 7 > /dev/null                                                     
./rocblas_sgemm_example -m2048  -n800   -k512  -a512    -b512   -c2048   -tTN -ov -fn
rocm-smi -c | grep "GPU Clock Level"                                                      
rocm-smi -d 0 --setsclk 7 > /dev/null                                                     
./rocblas_sgemm_example -m2048  -n1600  -k512  -a512    -b512   -c2048   -tTN -ov -fn
rocm-smi -c | grep "GPU Clock Level"                                                      
rocm-smi -d 0 --setsclk 7 > /dev/null                                                     
./rocblas_sgemm_example -m2048  -n3200  -k512  -a512    -b512   -c2048   -tTN -ov -fn
rocm-smi -c | grep "GPU Clock Level"                                                      
rocm-smi -d 0 --setsclk 7 > /dev/null                                                     
./rocblas_sgemm_example -m4096  -n400   -k1024 -a1024   -b1024  -c4096   -tTN -ov -fn
rocm-smi -c | grep "GPU Clock Level"                                                      
rocm-smi -d 0 --setsclk 7 > /dev/null                                                     
./rocblas_sgemm_example -m4096  -n800   -k1024 -a1024   -b1024  -c4096   -tTN -ov -fn
rocm-smi -c | grep "GPU Clock Level"                                                      
rocm-smi -d 0 --setsclk 7 > /dev/null                                                     
./rocblas_sgemm_example -m4096  -n1600  -k1024 -a1024   -b1024  -c4096   -tTN -ov -fn
rocm-smi -c | grep "GPU Clock Level"                                                      
rocm-smi -d 0 --setsclk 7 > /dev/null                                                     
./rocblas_sgemm_example -m4096  -n3200  -k1024 -a1024   -b1024  -c4096   -tTN -ov -fn
rocm-smi -c | grep "GPU Clock Level"                                                      
rocm-smi -d 0 --setsclk 7 > /dev/null                                                     
./rocblas_sgemm_example -m8192  -n400   -k2048 -a2048   -b2048  -c8192   -tTN -ov -fn
rocm-smi -c | grep "GPU Clock Level"                                                      
rocm-smi -d 0 --setsclk 7 > /dev/null                                                     
./rocblas_sgemm_example -m8192  -n800   -k2048 -a2048   -b2048  -c8192   -tTN -ov -fn
rocm-smi -c | grep "GPU Clock Level"                                                      
rocm-smi -d 0 --setsclk 7 > /dev/null                                                     
./rocblas_sgemm_example -m8192  -n1600  -k2048 -a2048   -b2048  -c8192   -tTN -ov -fn
rocm-smi -c | grep "GPU Clock Level"                                                      
rocm-smi -d 0 --setsclk 7 > /dev/null                                                     
./rocblas_sgemm_example -m8192  -n3200  -k2048 -a2048   -b2048  -c8192   -tTN -ov -fn
rocm-smi -c | grep "GPU Clock Level"                                                      
rocm-smi -d 0 --setsclk 7 > /dev/null                                                     
./rocblas_sgemm_example -m16384 -n400   -k4096 -a40964  -b4096  -c16384  -tTN -ov -fn
rocm-smi -c | grep "GPU Clock Level"                                                      
rocm-smi -d 0 --setsclk 7 > /dev/null                                                     
./rocblas_sgemm_example -m16384 -n800   -k4096 -a40964  -b4096  -c16384  -tTN -ov -fn
rocm-smi -c | grep "GPU Clock Level"                                                      
rocm-smi -d 0 --setsclk 7 > /dev/null                                                     
./rocblas_sgemm_example -m16384 -n1600  -k4096 -a40964  -b4096  -c16384  -tTN -ov -fn
rocm-smi -c | grep "GPU Clock Level"                                                      
rocm-smi -d 0 --setsclk 7 > /dev/null                                                     
./rocblas_sgemm_example -m16384 -n3200  -k4096 -a40964  -b4096  -c16384  -tTN -ov -fn
rocm-smi -c | grep "GPU Clock Level"                                                      
rocm-smi -d 0 --setsclk 7 > /dev/null                                                     
./rocblas_sgemm_example -m8448  -n48000 -k2816 -a2816   -b2816  -c8448   -tTN -ov -fn
rocm-smi -c | grep "GPU Clock Level"                                                      
rocm-smi -d 0 --setsclk 7 > /dev/null                                                     
./rocblas_sgemm_example -m8448  -n24000 -k2816 -a2816   -b2816  -c8448   -tTN -ov -fn
rocm-smi -c | grep "GPU Clock Level"                                                      
rocm-smi -d 0 --setsclk 7 > /dev/null                                                     
./rocblas_sgemm_example -m8448  -n12000 -k2816 -a2816   -b2816  -c8448   -tTN -ov -fn
rocm-smi -c | grep "GPU Clock Level"                                                      
rocm-smi -d 0 --setsclk 7 > /dev/null                                                     
./rocblas_sgemm_example -m8448  -n5984  -k2816 -a2816   -b2816  -c8448   -tTN -ov -fn
rocm-smi -c | grep "GPU Clock Level"                                                      
rocm-smi -d 0 --setsclk 7 > /dev/null                                                     
./rocblas_sgemm_example -m6144  -n48000 -k2048 -a2048   -b2048  -c6144   -tTN -ov -fn
rocm-smi -c | grep "GPU Clock Level"                                                      
rocm-smi -d 0 --setsclk 7 > /dev/null                                                     
./rocblas_sgemm_example -m6144  -n24000 -k2048 -a2048   -b2048  -c6144   -tTN -ov -fn
rocm-smi -c | grep "GPU Clock Level"                                                      
rocm-smi -d 0 --setsclk 7 > /dev/null                                                     
./rocblas_sgemm_example -m6144  -n12000 -k2048 -a2048   -b2048  -c6144   -tTN -ov -fn
rocm-smi -c | grep "GPU Clock Level"                                                      
rocm-smi -d 0 --setsclk 7 > /dev/null                                                     
./rocblas_sgemm_example -m6144  -n5984  -k2048 -a2048   -b2048  -c6144   -tTN -ov -fn
rocm-smi -c | grep "GPU Clock Level"                                                      
rocm-smi -d 0 --setsclk 7 > /dev/null                                                     
./rocblas_sgemm_example -m4608  -n48000 -k1536 -a1536   -b1536  -c4608   -tTN -ov -fn
rocm-smi -c | grep "GPU Clock Level"                                                      
rocm-smi -d 0 --setsclk 7 > /dev/null                                                     
./rocblas_sgemm_example -m4608  -n24000 -k1536 -a1536   -b1536  -c4608   -tTN -ov -fn
rocm-smi -c | grep "GPU Clock Level"                                                      
rocm-smi -d 0 --setsclk 7 > /dev/null                                                     
./rocblas_sgemm_example -m4608  -n12000 -k1536 -a1536   -b1536  -c4608   -tTN -ov -fn
rocm-smi -c | grep "GPU Clock Level"                                                      
rocm-smi -d 0 --setsclk 7 > /dev/null                                                     
./rocblas_sgemm_example -m4608  -n5984  -k1536 -a1536   -b1536  -c4608   -tTN -ov -fn
rocm-smi -c | grep "GPU Clock Level"                                                      
rocm-smi -d 0 --setsclk 7 > /dev/null                                                     
./rocblas_sgemm_example -m7680  -n48000 -k2560 -a2560   -b2560  -c7680   -tTN -ov -fn
rocm-smi -c | grep "GPU Clock Level"                                                      
rocm-smi -d 0 --setsclk 7 > /dev/null                                                     
./rocblas_sgemm_example -m7680  -n24000 -k2560 -a2560   -b2560  -c7680   -tTN -ov -fn
rocm-smi -c | grep "GPU Clock Level"                                                      
rocm-smi -d 0 --setsclk 7 > /dev/null                                                     
./rocblas_sgemm_example -m7680  -n12000 -k2560 -a2560   -b2560  -c7680   -tTN -ov -fn
rocm-smi -c | grep "GPU Clock Level"                                                      
rocm-smi -d 0 --setsclk 7 > /dev/null                                                     
./rocblas_sgemm_example -m7680  -n5984  -k2560 -a2560   -b2560  -c7680   -tTN -ov -fn
rocm-smi -c | grep "GPU Clock Level" 
rocm-smi -d 0 --setsclk 7 > /dev/null



rocm-smi -d 0 -g
rocm-smi -d 0 --setfan 50

