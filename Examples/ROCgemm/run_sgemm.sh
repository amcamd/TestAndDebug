#!/bin/sh

rocm-smi -d 0 --setsclk 6
sleep 2
rocm-smi -d 0 -g

#./rocblas_sgemm_example -m128 -n128 -k128 -a128 -b128 -c128 -tNT -ov -fy
#rocm-smi -c | grep "GPU Clock Level" 
#rocm-smi -d 0 --setsclk 6 > /dev/null
#./rocblas_sgemm_example -m256 -n256 -k256 -a256 -b256 -c256 -tNT -ov -fn
#rocm-smi -c | grep "GPU Clock Level" 
#rocm-smi -d 0 --setsclk 6 > /dev/null
#./rocblas_sgemm_example -m384 -n384 -k384 -a384 -b384 -c384 -tNT -ov -fn
#rocm-smi -c | grep "GPU Clock Level" 
#rocm-smi -d 0 --setsclk 6 > /dev/null
#./rocblas_sgemm_example -m512 -n512 -k512 -a512 -b512 -c512 -tNT -ov -fn
#rocm-smi -c | grep "GPU Clock Level" 
#rocm-smi -d 0 --setsclk 6 > /dev/null
#./rocblas_sgemm_example -m640 -n640 -k640 -a640 -b640 -c640 -tNT -ov -fn
#rocm-smi -c | grep "GPU Clock Level" 
#rocm-smi -d 0 --setsclk 6 > /dev/null
#./rocblas_sgemm_example -m768 -n768 -k768 -a768 -b768 -c768 -tNT -ov -fn
#rocm-smi -c | grep "GPU Clock Level" 
#rocm-smi -d 0 --setsclk 6 > /dev/null
#./rocblas_sgemm_example -m896 -n896 -k896 -a896 -b896 -c896 -tNT -ov -fn
#rocm-smi -c | grep "GPU Clock Level" 
#rocm-smi -d 0 --setsclk 6 > /dev/null
#./rocblas_sgemm_example -m1024 -n1024 -k1024 -a1024 -b1024 -c1024 -tNT -ov -fn
#rocm-smi -c | grep "GPU Clock Level" 
#rocm-smi -d 0 --setsclk 6 > /dev/null
#./rocblas_sgemm_example -m1152 -n1152 -k1152 -a1152 -b1152 -c1152 -tNT -ov -fn
#rocm-smi -c | grep "GPU Clock Level" 
#rocm-smi -d 0 --setsclk 6 > /dev/null
#./rocblas_sgemm_example -m1280 -n1280 -k1280 -a1280 -b1280 -c1280 -tNT -ov -fn
#rocm-smi -c | grep "GPU Clock Level" 
#rocm-smi -d 0 --setsclk 6 > /dev/null
#./rocblas_sgemm_example -m1408 -n1408 -k1408 -a1408 -b1408 -c1408 -tNT -ov -fn
#rocm-smi -c | grep "GPU Clock Level" 
#rocm-smi -d 0 --setsclk 6 > /dev/null
#./rocblas_sgemm_example -m1536 -n1536 -k1536 -a1536 -b1536 -c1536 -tNT -ov -fn
#rocm-smi -c | grep "GPU Clock Level" 
#rocm-smi -d 0 --setsclk 6 > /dev/null
#./rocblas_sgemm_example -m1664 -n1664 -k1664 -a1664 -b1664 -c1664 -tNT -ov -fn
#rocm-smi -c | grep "GPU Clock Level" 
#rocm-smi -d 0 --setsclk 6 > /dev/null
#./rocblas_sgemm_example -m1792 -n1792 -k1792 -a1792 -b1792 -c1792 -tNT -ov -fn
#rocm-smi -c | grep "GPU Clock Level" 
#rocm-smi -d 0 --setsclk 6 > /dev/null
#./rocblas_sgemm_example -m1920 -n1920 -k1920 -a1920 -b1920 -c1920 -tNT -ov -fn
#rocm-smi -c | grep "GPU Clock Level" 
#rocm-smi -d 0 --setsclk 6 > /dev/null
#./rocblas_sgemm_example -m2048 -n2048 -k2048 -a2048 -b2048 -c2048 -tNT -ov -fn
#rocm-smi -c | grep "GPU Clock Level" 
#rocm-smi -d 0 --setsclk 6 > /dev/null
#./rocblas_sgemm_example -m2176 -n2176 -k2176 -a2176 -b2176 -c2176 -tNT -ov -fn
#rocm-smi -c | grep "GPU Clock Level" 
#rocm-smi -d 0 --setsclk 6 > /dev/null
#./rocblas_sgemm_example -m2304 -n2304 -k2304 -a2304 -b2304 -c2304 -tNT -ov -fn
#rocm-smi -c | grep "GPU Clock Level" 
#rocm-smi -d 0 --setsclk 6 > /dev/null
#./rocblas_sgemm_example -m2432 -n2432 -k2432 -a2432 -b2432 -c2432 -tNT -ov -fn
#rocm-smi -c | grep "GPU Clock Level" 
#rocm-smi -d 0 --setsclk 6 > /dev/null
#./rocblas_sgemm_example -m2560 -n2560 -k2560 -a2560 -b2560 -c2560 -tNT -ov -fn
#rocm-smi -c | grep "GPU Clock Level" 
#rocm-smi -d 0 --setsclk 6 > /dev/null
#./rocblas_sgemm_example -m2688 -n2688 -k2688 -a2688 -b2688 -c2688 -tNT -ov -fn
#rocm-smi -c | grep "GPU Clock Level" 
#rocm-smi -d 0 --setsclk 6 > /dev/null
#./rocblas_sgemm_example -m2816 -n2816 -k2816 -a2816 -b2816 -c2816 -tNT -ov -fn
#rocm-smi -c | grep "GPU Clock Level" 
#rocm-smi -d 0 --setsclk 6 > /dev/null
#./rocblas_sgemm_example -m2944 -n2944 -k2944 -a2944 -b2944 -c2944 -tNT -ov -fn
#rocm-smi -c | grep "GPU Clock Level" 
#rocm-smi -d 0 --setsclk 6 > /dev/null
#./rocblas_sgemm_example -m3072 -n3072 -k3072 -a3072 -b3072 -c3072 -tNT -ov -fn
#rocm-smi -c | grep "GPU Clock Level" 
#rocm-smi -d 0 --setsclk 6 > /dev/null
#./rocblas_sgemm_example -m3200 -n3200 -k3200 -a3200 -b3200 -c3200 -tNT -ov -fn
#rocm-smi -c | grep "GPU Clock Level" 
#rocm-smi -d 0 --setsclk 6 > /dev/null
#./rocblas_sgemm_example -m3328 -n3328 -k3328 -a3328 -b3328 -c3328 -tNT -ov -fn
#rocm-smi -c | grep "GPU Clock Level" 
#rocm-smi -d 0 --setsclk 6 > /dev/null
#./rocblas_sgemm_example -m3456 -n3456 -k3456 -a3456 -b3456 -c3456 -tNT -ov -fn
#rocm-smi -c | grep "GPU Clock Level" 
#rocm-smi -d 0 --setsclk 6 > /dev/null
#./rocblas_sgemm_example -m3584 -n3584 -k3584 -a3584 -b3584 -c3584 -tNT -ov -fn
#rocm-smi -c | grep "GPU Clock Level" 
#rocm-smi -d 0 --setsclk 6 > /dev/null
#./rocblas_sgemm_example -m3712 -n3712 -k3712 -a3712 -b3712 -c3712 -tNT -ov -fn
#rocm-smi -c | grep "GPU Clock Level" 
#rocm-smi -d 0 --setsclk 6 > /dev/null
#./rocblas_sgemm_example -m3804 -n3804 -k3804 -a3804 -b3804 -c3804 -tNT -ov -fn
#rocm-smi -c | grep "GPU Clock Level" 
#rocm-smi -d 0 --setsclk 6 > /dev/null
#./rocblas_sgemm_example -m3968 -n3968 -k3968 -a3968 -b3968 -c3968 -tNT -ov -fn
#rocm-smi -c | grep "GPU Clock Level" 
#rocm-smi -d 0 --setsclk 6 > /dev/null
#./rocblas_sgemm_example -m4096 -n4096 -k4096 -a4096 -b4096 -c4096 -tNT -ov -fn
#rocm-smi -c | grep "GPU Clock Level" 
#rocm-smi -d 0 --setsclk 6 > /dev/null
#./rocblas_sgemm_example -m4224 -n4224 -k4224 -a4224 -b4224 -c4224 -tNT -ov -fn
#rocm-smi -c | grep "GPU Clock Level" 
#rocm-smi -d 0 --setsclk 6 > /dev/null
#./rocblas_sgemm_example -m4352 -n4352 -k4352 -a4352 -b4352 -c4352 -tNT -ov -fn
#rocm-smi -c | grep "GPU Clock Level" 
#rocm-smi -d 0 --setsclk 6 > /dev/null
#./rocblas_sgemm_example -m4480 -n4480 -k4480 -a4480 -b4480 -c4480 -tNT -ov -fn
#rocm-smi -c | grep "GPU Clock Level" 
#rocm-smi -d 0 --setsclk 6 > /dev/null
#./rocblas_sgemm_example -m4608 -n4608 -k4608 -a4608 -b4608 -c4608 -tNT -ov -fn
#rocm-smi -c | grep "GPU Clock Level" 
#rocm-smi -d 0 --setsclk 6 > /dev/null
#./rocblas_sgemm_example -m4736 -n4736 -k4736 -a4736 -b4736 -c4736 -tNT -ov -fn
#rocm-smi -c | grep "GPU Clock Level" 
#rocm-smi -d 0 --setsclk 6 > /dev/null
#./rocblas_sgemm_example -m4864 -n4864 -k4864 -a4864 -b4864 -c4864 -tNT -ov -fn
#rocm-smi -c | grep "GPU Clock Level" 
#rocm-smi -d 0 --setsclk 6 > /dev/null
#./rocblas_sgemm_example -m4992 -n4992 -k4992 -a4992 -b4992 -c4992 -tNT -ov -fn
#rocm-smi -c | grep "GPU Clock Level" 
#rocm-smi -d 0 --setsclk 6 > /dev/null
#./rocblas_sgemm_example -m5120 -n5120 -k5120 -a5120 -b5120 -c5120 -tNT -ov -fn
#rocm-smi -c | grep "GPU Clock Level" 


rocm-smi -d 0 --setsclk 6 > /dev/null
./rocblas_sgemm_example -m5248 -n5248 -k5248 -a5248 -b5248 -c5248 -tNT -ov -fn
rocm-smi -c | grep "GPU Clock Level" 
rocm-smi -d 0 --setsclk 6 > /dev/null
./rocblas_sgemm_example -m5376 -n5376 -k5376 -a5376 -b5376 -c5376 -tNT -ov -fn
rocm-smi -c | grep "GPU Clock Level" 
rocm-smi -d 0 --setsclk 6 > /dev/null
./rocblas_sgemm_example -m5504 -n5504 -k5504 -a5504 -b5504 -c5504 -tNT -ov -fn
rocm-smi -c | grep "GPU Clock Level" 
rocm-smi -d 0 --setsclk 6 > /dev/null
./rocblas_sgemm_example -m5632 -n5632 -k5632 -a5632 -b5632 -c5632 -tNT -ov -fn
rocm-smi -c | grep "GPU Clock Level" 
rocm-smi -d 0 --setsclk 6 > /dev/null
./rocblas_sgemm_example -m5760 -n5760 -k5760 -a5760 -b5760 -c5760 -tNT -ov -fn
rocm-smi -c | grep "GPU Clock Level" 
rocm-smi -d 0 --setsclk 6 > /dev/null
./rocblas_sgemm_example -m5888 -n5888 -k5888 -a5888 -b5888 -c5888 -tNT -ov -fn
rocm-smi -c | grep "GPU Clock Level" 
rocm-smi -d 0 --setsclk 6 > /dev/null
./rocblas_sgemm_example -m6016 -n6016 -k6016 -a6016 -b6016 -c6016 -tNT -ov -fn
rocm-smi -c | grep "GPU Clock Level" 
rocm-smi -d 0 --setsclk 6 > /dev/null
./rocblas_sgemm_example -m6144 -n6144 -k6144 -a6144 -b6144 -c6144 -tNT -ov -fn
rocm-smi -c | grep "GPU Clock Level" 
rocm-smi -d 0 --setsclk 6 > /dev/null
./rocblas_sgemm_example -m6272 -n6272 -k6272 -a6272 -b6272 -c6272 -tNT -ov -fn
rocm-smi -c | grep "GPU Clock Level" 
rocm-smi -d 0 --setsclk 6 > /dev/null
./rocblas_sgemm_example -m6400 -n6400 -k6400 -a6400 -b6400 -c6400 -tNT -ov -fn
rocm-smi -c | grep "GPU Clock Level" 
rocm-smi -d 0 --setsclk 6 > /dev/null
./rocblas_sgemm_example -m6528 -n6528 -k6528 -a6528 -b6528 -c6528 -tNT -ov -fn
rocm-smi -c | grep "GPU Clock Level" 
rocm-smi -d 0 --setsclk 6 > /dev/null
./rocblas_sgemm_example -m6656 -n6656 -k6656 -a6656 -b6656 -c6656 -tNT -ov -fn
rocm-smi -c | grep "GPU Clock Level" 
rocm-smi -d 0 --setsclk 6 > /dev/null
./rocblas_sgemm_example -m6784 -n6784 -k6784 -a6784 -b6784 -c6784 -tNT -ov -fn
rocm-smi -c | grep "GPU Clock Level" 
rocm-smi -d 0 --setsclk 6 > /dev/null
./rocblas_sgemm_example -m6912 -n6912 -k6912 -a6912 -b6912 -c6912 -tNT -ov -fn
rocm-smi -c | grep "GPU Clock Level" 
rocm-smi -d 0 --setsclk 6 > /dev/null
./rocblas_sgemm_example -m7040 -n7040 -k7040 -a7040 -b7040 -c7040 -tNT -ov -fn
rocm-smi -c | grep "GPU Clock Level" 
rocm-smi -d 0 --setsclk 6 > /dev/null
./rocblas_sgemm_example -m7168 -n7168 -k7168 -a7168 -b7168 -c7168 -tNT -ov -fn
rocm-smi -c | grep "GPU Clock Level" 
rocm-smi -d 0 --setsclk 6 > /dev/null
./rocblas_sgemm_example -m7296 -n7296 -k7296 -a7296 -b7296 -c7296 -tNT -ov -fn
rocm-smi -c | grep "GPU Clock Level" 
rocm-smi -d 0 --setsclk 6 > /dev/null
./rocblas_sgemm_example -m7424 -n7424 -k7424 -a7424 -b7424 -c7424 -tNT -ov -fn
rocm-smi -c | grep "GPU Clock Level" 
rocm-smi -d 0 --setsclk 6 > /dev/null
./rocblas_sgemm_example -m7552 -n7552 -k7552 -a7552 -b7552 -c7552 -tNT -ov -fn
rocm-smi -c | grep "GPU Clock Level" 
rocm-smi -d 0 --setsclk 6 > /dev/null
./rocblas_sgemm_example -m7680 -n7680 -k7680 -a7680 -b7680 -c7680 -tNT -ov -fn
rocm-smi -c | grep "GPU Clock Level" 
rocm-smi -d 0 --setsclk 6 > /dev/null
./rocblas_sgemm_example -m7808 -n7808 -k7808 -a7808 -b7808 -c7808 -tNT -ov -fn
rocm-smi -c | grep "GPU Clock Level" 
rocm-smi -d 0 --setsclk 6 > /dev/null
./rocblas_sgemm_example -m7936 -n7936 -k7936 -a7936 -b7936 -c7936 -tNT -ov -fn
rocm-smi -c | grep "GPU Clock Level" 
rocm-smi -d 0 --setsclk 6 > /dev/null
./rocblas_sgemm_example -m8064 -n8064 -k8064 -a8064 -b8064 -c8064 -tNT -ov -fn
rocm-smi -c | grep "GPU Clock Level" 
rocm-smi -d 0 --setsclk 6 > /dev/null
./rocblas_sgemm_example -m8192 -n8192 -k8192 -a8192 -b8192 -c8192 -tNT -ov -fn
rocm-smi -c | grep "GPU Clock Level" 
rocm-smi -d 0 --setsclk 6 > /dev/null
./rocblas_sgemm_example -m8320 -n8320 -k8320 -a8320 -b8320 -c8320 -tNT -ov -fn
rocm-smi -c | grep "GPU Clock Level" 
rocm-smi -d 0 --setsclk 6 > /dev/null
./rocblas_sgemm_example -m8448 -n8448 -k8448 -a8448 -b8448 -c8448 -tNT -ov -fn
rocm-smi -c | grep "GPU Clock Level" 
rocm-smi -d 0 --setsclk 6 > /dev/null
./rocblas_sgemm_example -m8576 -n8576 -k8576 -a8576 -b8576 -c8576 -tNT -ov -fn
rocm-smi -c | grep "GPU Clock Level" 
rocm-smi -d 0 --setsclk 6 > /dev/null
./rocblas_sgemm_example -m8704 -n8704 -k8704 -a8704 -b8704 -c8704 -tNT -ov -fn
rocm-smi -c | grep "GPU Clock Level" 
rocm-smi -d 0 --setsclk 6 > /dev/null
./rocblas_sgemm_example -m8832 -n8832 -k8832 -a8832 -b8832 -c8832 -tNT -ov -fn
rocm-smi -c | grep "GPU Clock Level" 
rocm-smi -d 0 --setsclk 6 > /dev/null
./rocblas_sgemm_example -m8192 -n8192 -k8192 -a8192 -b8192 -c8192 -tNT -ov -fn
rocm-smi -c | grep "GPU Clock Level" 
rocm-smi -d 0 --setsclk 6 > /dev/null
./rocblas_sgemm_example -m8960 -n8960 -k8960 -a8960 -b8960 -c8960 -tNT -ov -fn
rocm-smi -c | grep "GPU Clock Level" 
rocm-smi -d 0 --setsclk 6 > /dev/null
./rocblas_sgemm_example -m9088 -n9088 -k9088 -a9088 -b9088 -c9088 -tNT -ov -fn
rocm-smi -c | grep "GPU Clock Level" 
rocm-smi -d 0 --setsclk 6 > /dev/null
./rocblas_sgemm_example -m9216 -n9216 -k9216 -a9216 -b9216 -c9216 -tNT -ov -fn
rocm-smi -c | grep "GPU Clock Level" 
rocm-smi -d 0 --setsclk 6 > /dev/null
./rocblas_sgemm_example -m9344 -n9344 -k9344 -a9344 -b9344 -c9344 -tNT -ov -fn
rocm-smi -c | grep "GPU Clock Level" 
rocm-smi -d 0 --setsclk 6 > /dev/null
./rocblas_sgemm_example -m9472 -n9472 -k9472 -a9472 -b9472 -c9472 -tNT -ov -fn
rocm-smi -c | grep "GPU Clock Level" 
rocm-smi -d 0 --setsclk 6 > /dev/null
./rocblas_sgemm_example -m9600 -n9600 -k9600 -a9600 -b9600 -c9600 -tNT -ov -fn

rocm-smi -d 0 -g
