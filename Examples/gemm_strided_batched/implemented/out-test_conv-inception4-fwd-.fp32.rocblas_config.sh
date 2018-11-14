./rocblas-bench -f gemm -r s --transposeA N --transposeB N -m 1225 -n 192 -k 1728 --alpha 1 --lda 1225 --ldb 1728 --beta 0 --ldc 1225 >> gemm.txt
./rocblas-bench -f gemm -r s --transposeA N --transposeB N -m 1225 -n 224 -k 1728 --alpha 1 --lda 1225 --ldb 1728 --beta 0 --ldc 1225 >> gemm.txt
./rocblas-bench -f gemm -r s --transposeA N --transposeB N -m 1225 -n 96 -k 576 --alpha 1 --lda 1225 --ldb 576 --beta 0 --ldc 1225 >> gemm.txt
./rocblas-bench -f gemm -r s --transposeA N --transposeB N -m 1225 -n 96 -k 864 --alpha 1 --lda 1225 --ldb 864 --beta 0 --ldc 1225 >> gemm.txt
./rocblas-bench -f gemm -r s --transposeA N --transposeB N -m 2048 -n 256 -k 1536 --alpha 1 --lda 2048 --ldb 1536 --beta 0 --ldc 2048 >> gemm.txt
./rocblas-bench -f gemm -r s --transposeA N --transposeB N -m 2048 -n 384 -k 1536 --alpha 1 --lda 2048 --ldb 1536 --beta 0 --ldc 2048 >> gemm.txt
./rocblas-bench -f gemm -r s --transposeA N --transposeB N -m 21609 -n 32 -k 288 --alpha 1 --lda 21609 --ldb 288 --beta 0 --ldc 21609 >> gemm.txt
./rocblas-bench -f gemm -r s --transposeA N --transposeB N -m 21609 -n 64 -k 288 --alpha 1 --lda 21609 --ldb 288 --beta 0 --ldc 21609 >> gemm.txt
./rocblas-bench -f gemm -r s --transposeA N --transposeB N -m 22201 -n 32 -k 27 --alpha 1 --lda 22201 --ldb 27 --beta 0 --ldc 22201 >> gemm.txt
./rocblas-bench -f gemm -r s --transposeA N --transposeB N -m 289 -n 192 -k 1344 --alpha 1 --lda 289 --ldb 1344 --beta 0 --ldc 289 >> gemm.txt
./rocblas-bench -f gemm -r s --transposeA N --transposeB N -m 289 -n 224 -k 1344 --alpha 1 --lda 289 --ldb 1344 --beta 0 --ldc 289 >> gemm.txt
./rocblas-bench -f gemm -r s --transposeA N --transposeB N -m 289 -n 224 -k 1568 --alpha 1 --lda 289 --ldb 1568 --beta 0 --ldc 289 >> gemm.txt
./rocblas-bench -f gemm -r s --transposeA N --transposeB N -m 289 -n 256 -k 1568 --alpha 1 --lda 289 --ldb 1568 --beta 0 --ldc 289 >> gemm.txt
./rocblas-bench -f gemm -r s --transposeA N --transposeB N -m 289 -n 256 -k 1792 --alpha 1 --lda 289 --ldb 1792 --beta 0 --ldc 289 >> gemm.txt
./rocblas-bench -f gemm -r s --transposeA N --transposeB N -m 289 -n 256 -k 2016 --alpha 1 --lda 289 --ldb 2016 --beta 0 --ldc 289 >> gemm.txt
./rocblas-bench -f gemm -r s --transposeA N --transposeB N -m 289 -n 320 -k 1792 --alpha 1 --lda 289 --ldb 1792 --beta 0 --ldc 289 >> gemm.txt
./rocblas-bench -f gemm -r s --transposeA N --transposeB N -m 289 -n 384 -k 3456 --alpha 1 --lda 289 --ldb 3456 --beta 0 --ldc 289 >> gemm.txt
./rocblas-bench -f gemm -r s --transposeA N --transposeB N -m 5041 -n 96 -k 576 --alpha 1 --lda 5041 --ldb 576 --beta 0 --ldc 5041 >> gemm.txt
./rocblas-bench -f gemm -r s --transposeA N --transposeB N -m 5329 -n 64 -k 448 --alpha 1 --lda 5329 --ldb 448 --beta 0 --ldc 5329 >> gemm.txt
./rocblas-bench -f gemm -r s --transposeA N --transposeB N -m 5329 -n 96 -k 576 --alpha 1 --lda 5329 --ldb 576 --beta 0 --ldc 5329 >> gemm.txt
./rocblas-bench -f gemm -r s --transposeA N --transposeB N -m 64 -n 192 -k 1728 --alpha 1 --lda 64 --ldb 1728 --beta 0 --ldc 64 >> gemm.txt
./rocblas-bench -f gemm -r s --transposeA N --transposeB N -m 64 -n 256 -k 1152 --alpha 1 --lda 64 --ldb 1152 --beta 0 --ldc 64 >> gemm.txt
./rocblas-bench -f gemm -r s --transposeA N --transposeB N -m 64 -n 256 -k 1536 --alpha 1 --lda 64 --ldb 1536 --beta 0 --ldc 64 >> gemm.txt
./rocblas-bench -f gemm -r s --transposeA N --transposeB N -m 64 -n 320 -k 2880 --alpha 1 --lda 64 --ldb 2880 --beta 0 --ldc 64 >> gemm.txt
./rocblas-bench -f gemm -r s --transposeA N --transposeB N -m 64 -n 448 -k 1152 --alpha 1 --lda 64 --ldb 1152 --beta 0 --ldc 64 >> gemm.txt
./rocblas-bench -f gemm -r s --transposeA N --transposeB N -m 64 -n 512 -k 1344 --alpha 1 --lda 64 --ldb 1344 --beta 0 --ldc 64 >> gemm.txt
./rocblas-bench -f gemm_strided_batched -r s --transposeA N --transposeB N -m 1225 -n 192 -k 384 --alpha 1 --lda 1225 --stride_a 470400 --ldb 384 --stride_b 0 --beta 0 --ldc 1225 --stride_c 235200 --batch 32 >> gemm.sb.txt
./rocblas-bench -f gemm_strided_batched -r s --transposeA N --transposeB N -m 1225 -n 64 -k 384 --alpha 1 --lda 1225 --stride_a 470400 --ldb 384 --stride_b 0 --beta 0 --ldc 1225 --stride_c 78400 --batch 32 >> gemm.sb.txt
./rocblas-bench -f gemm_strided_batched -r s --transposeA N --transposeB N -m 1225 -n 96 -k 384 --alpha 1 --lda 1225 --stride_a 470400 --ldb 384 --stride_b 0 --beta 0 --ldc 1225 --stride_c 117600 --batch 32 >> gemm.sb.txt
./rocblas-bench -f gemm_strided_batched -r s --transposeA N --transposeB N -m 289 -n 128 -k 1024 --alpha 1 --lda 289 --stride_a 295936 --ldb 1024 --stride_b 0 --beta 0 --ldc 289 --stride_c 36992 --batch 32 >> gemm.sb.txt
./rocblas-bench -f gemm_strided_batched -r s --transposeA N --transposeB N -m 289 -n 192 -k 1024 --alpha 1 --lda 289 --stride_a 295936 --ldb 1024 --stride_b 0 --beta 0 --ldc 289 --stride_c 55488 --batch 32 >> gemm.sb.txt
./rocblas-bench -f gemm_strided_batched -r s --transposeA N --transposeB N -m 289 -n 256 -k 1024 --alpha 1 --lda 289 --stride_a 295936 --ldb 1024 --stride_b 0 --beta 0 --ldc 289 --stride_c 73984 --batch 32 >> gemm.sb.txt
./rocblas-bench -f gemm_strided_batched -r s --transposeA N --transposeB N -m 289 -n 384 -k 1024 --alpha 1 --lda 289 --stride_a 295936 --ldb 1024 --stride_b 0 --beta 0 --ldc 289 --stride_c 110976 --batch 32 >> gemm.sb.txt
./rocblas-bench -f gemm_strided_batched -r s --transposeA N --transposeB N -m 5329 -n 64 -k 160 --alpha 1 --lda 5329 --stride_a 852640 --ldb 160 --stride_b 0 --beta 0 --ldc 5329 --stride_c 341056 --batch 32 >> gemm.sb.txt
