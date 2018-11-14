./rocblas-bench -f gemm -r s --transposeA N --transposeB T -m 1225 -n 1728 -k 192 --alpha 1 --lda 1225 --ldb 1728 --beta 0 --ldc 1225 >> gemm.txt
./rocblas-bench -f gemm -r s --transposeA N --transposeB T -m 1225 -n 1728 -k 224 --alpha 1 --lda 1225 --ldb 1728 --beta 0 --ldc 1225 >> gemm.txt
./rocblas-bench -f gemm -r s --transposeA N --transposeB T -m 1225 -n 576 -k 96 --alpha 1 --lda 1225 --ldb 576 --beta 0 --ldc 1225 >> gemm.txt
./rocblas-bench -f gemm -r s --transposeA N --transposeB T -m 1225 -n 864 -k 96 --alpha 1 --lda 1225 --ldb 864 --beta 0 --ldc 1225 >> gemm.txt
./rocblas-bench -f gemm -r s --transposeA N --transposeB T -m 21609 -n 288 -k 32 --alpha 1 --lda 21609 --ldb 288 --beta 0 --ldc 21609 >> gemm.txt
./rocblas-bench -f gemm -r s --transposeA N --transposeB T -m 21609 -n 288 -k 64 --alpha 1 --lda 21609 --ldb 288 --beta 0 --ldc 21609 >> gemm.txt
./rocblas-bench -f gemm -r s --transposeA N --transposeB T -m 22201 -n 27 -k 32 --alpha 1 --lda 22201 --ldb 27 --beta 0 --ldc 22201 >> gemm.txt
./rocblas-bench -f gemm -r s --transposeA N --transposeB T -m 289 -n 1344 -k 192 --alpha 1 --lda 289 --ldb 1344 --beta 0 --ldc 289 >> gemm.txt
./rocblas-bench -f gemm -r s --transposeA N --transposeB T -m 289 -n 1344 -k 224 --alpha 1 --lda 289 --ldb 1344 --beta 0 --ldc 289 >> gemm.txt
./rocblas-bench -f gemm -r s --transposeA N --transposeB T -m 289 -n 1568 -k 224 --alpha 1 --lda 289 --ldb 1568 --beta 0 --ldc 289 >> gemm.txt
./rocblas-bench -f gemm -r s --transposeA N --transposeB T -m 289 -n 1568 -k 256 --alpha 1 --lda 289 --ldb 1568 --beta 0 --ldc 289 >> gemm.txt
./rocblas-bench -f gemm -r s --transposeA N --transposeB T -m 289 -n 1792 -k 256 --alpha 1 --lda 289 --ldb 1792 --beta 0 --ldc 289 >> gemm.txt
./rocblas-bench -f gemm -r s --transposeA N --transposeB T -m 289 -n 1792 -k 320 --alpha 1 --lda 289 --ldb 1792 --beta 0 --ldc 289 >> gemm.txt
./rocblas-bench -f gemm -r s --transposeA N --transposeB T -m 289 -n 2016 -k 256 --alpha 1 --lda 289 --ldb 2016 --beta 0 --ldc 289 >> gemm.txt
./rocblas-bench -f gemm -r s --transposeA N --transposeB T -m 289 -n 3456 -k 384 --alpha 1 --lda 289 --ldb 3456 --beta 0 --ldc 289 >> gemm.txt
./rocblas-bench -f gemm -r s --transposeA N --transposeB T -m 5041 -n 576 -k 96 --alpha 1 --lda 5041 --ldb 576 --beta 0 --ldc 5041 >> gemm.txt
./rocblas-bench -f gemm -r s --transposeA N --transposeB T -m 5329 -n 448 -k 64 --alpha 1 --lda 5329 --ldb 448 --beta 0 --ldc 5329 >> gemm.txt
./rocblas-bench -f gemm -r s --transposeA N --transposeB T -m 5329 -n 576 -k 96 --alpha 1 --lda 5329 --ldb 576 --beta 0 --ldc 5329 >> gemm.txt
./rocblas-bench -f gemm -r s --transposeA N --transposeB T -m 64 -n 1152 -k 256 --alpha 1 --lda 64 --ldb 1152 --beta 0 --ldc 64 >> gemm.txt
./rocblas-bench -f gemm -r s --transposeA N --transposeB T -m 64 -n 1152 -k 448 --alpha 1 --lda 64 --ldb 1152 --beta 0 --ldc 64 >> gemm.txt
./rocblas-bench -f gemm -r s --transposeA N --transposeB T -m 64 -n 1344 -k 512 --alpha 1 --lda 64 --ldb 1344 --beta 0 --ldc 64 >> gemm.txt
./rocblas-bench -f gemm -r s --transposeA N --transposeB T -m 64 -n 1536 -k 256 --alpha 1 --lda 64 --ldb 1536 --beta 0 --ldc 64 >> gemm.txt
./rocblas-bench -f gemm -r s --transposeA N --transposeB T -m 64 -n 1728 -k 192 --alpha 1 --lda 64 --ldb 1728 --beta 0 --ldc 64 >> gemm.txt
./rocblas-bench -f gemm -r s --transposeA N --transposeB T -m 64 -n 2880 -k 320 --alpha 1 --lda 64 --ldb 2880 --beta 0 --ldc 64 >> gemm.txt
./rocblas-bench -f gemm_strided_batched -r s --transposeA N --transposeB T -m 1225 -n 384 -k 192 --alpha 1 --lda 1225 --stride_a 235200 --ldb 384 --stride_b 0 --beta 0 --ldc 1225 --stride_c 470400 --batch 32 >> gemm.sb.txt
./rocblas-bench -f gemm_strided_batched -r s --transposeA N --transposeB T -m 1225 -n 384 -k 64 --alpha 1 --lda 1225 --stride_a 78400 --ldb 384 --stride_b 0 --beta 0 --ldc 1225 --stride_c 470400 --batch 32 >> gemm.sb.txt
./rocblas-bench -f gemm_strided_batched -r s --transposeA N --transposeB T -m 1225 -n 384 -k 96 --alpha 1 --lda 1225 --stride_a 117600 --ldb 384 --stride_b 0 --beta 0 --ldc 1225 --stride_c 470400 --batch 32 >> gemm.sb.txt
./rocblas-bench -f gemm_strided_batched -r s --transposeA N --transposeB T -m 289 -n 1024 -k 128 --alpha 1 --lda 289 --stride_a 36992 --ldb 1024 --stride_b 0 --beta 0 --ldc 289 --stride_c 295936 --batch 32 >> gemm.sb.txt
./rocblas-bench -f gemm_strided_batched -r s --transposeA N --transposeB T -m 289 -n 1024 -k 192 --alpha 1 --lda 289 --stride_a 55488 --ldb 1024 --stride_b 0 --beta 0 --ldc 289 --stride_c 295936 --batch 32 >> gemm.sb.txt
./rocblas-bench -f gemm_strided_batched -r s --transposeA N --transposeB T -m 289 -n 1024 -k 256 --alpha 1 --lda 289 --stride_a 73984 --ldb 1024 --stride_b 0 --beta 0 --ldc 289 --stride_c 295936 --batch 32 >> gemm.sb.txt
./rocblas-bench -f gemm_strided_batched -r s --transposeA N --transposeB T -m 289 -n 1024 -k 384 --alpha 1 --lda 289 --stride_a 110976 --ldb 1024 --stride_b 0 --beta 0 --ldc 289 --stride_c 295936 --batch 32 >> gemm.sb.txt
./rocblas-bench -f gemm_strided_batched -r s --transposeA N --transposeB T -m 5329 -n 160 -k 64 --alpha 1 --lda 5329 --stride_a 341056 --ldb 160 --stride_b 0 --beta 0 --ldc 5329 --stride_c 852640 --batch 32 >> gemm.sb.txt
./rocblas-bench -f gemm_strided_batched -r s --transposeA N --transposeB T -m 64 -n 1536 -k 256 --alpha 1 --lda 64 --stride_a 16384 --ldb 1536 --stride_b 0 --beta 0 --ldc 64 --stride_c 98304 --batch 32 >> gemm.sb.txt
./rocblas-bench -f gemm_strided_batched -r s --transposeA N --transposeB T -m 64 -n 1536 -k 384 --alpha 1 --lda 64 --stride_a 24576 --ldb 1536 --stride_b 0 --beta 0 --ldc 64 --stride_c 98304 --batch 32 >> gemm.sb.txt
