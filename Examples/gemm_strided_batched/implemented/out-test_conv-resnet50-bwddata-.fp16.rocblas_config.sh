./rocblas-bench -f gemm -r h --transposeA N --transposeB T -m 12544 -n 147 -k 64 --alpha 15360 --lda 12544 --ldb 147 --beta 0 --ldc 12544 >> gemm.txt
./rocblas-bench -f gemm -r h --transposeA N --transposeB T -m 12544 -n 512 -k 1024 --alpha 15360 --lda 12544 --ldb 512 --beta 0 --ldc 12544 >> gemm.txt
./rocblas-bench -f gemm -r h --transposeA N --transposeB T -m 12544 -n 512 -k 256 --alpha 15360 --lda 12544 --ldb 512 --beta 0 --ldc 12544 >> gemm.txt
./rocblas-bench -f gemm -r h --transposeA N --transposeB T -m 196 -n 2304 -k 256 --alpha 15360 --lda 196 --ldb 2304 --beta 0 --ldc 196 >> gemm.txt
./rocblas-bench -f gemm -r h --transposeA N --transposeB T -m 3025 -n 576 -k 64 --alpha 15360 --lda 3025 --ldb 576 --beta 0 --ldc 3025 >> gemm.txt
./rocblas-bench -f gemm -r h --transposeA N --transposeB T -m 3136 -n 1024 -k 2048 --alpha 15360 --lda 3136 --ldb 1024 --beta 0 --ldc 3136 >> gemm.txt
./rocblas-bench -f gemm -r h --transposeA N --transposeB T -m 3136 -n 1024 -k 512 --alpha 15360 --lda 3136 --ldb 1024 --beta 0 --ldc 3136 >> gemm.txt
./rocblas-bench -f gemm -r h --transposeA N --transposeB T -m 3136 -n 576 -k 64 --alpha 15360 --lda 3136 --ldb 576 --beta 0 --ldc 3136 >> gemm.txt
./rocblas-bench -f gemm -r h --transposeA N --transposeB T -m 49 -n 4608 -k 512 --alpha 15360 --lda 49 --ldb 4608 --beta 0 --ldc 49 >> gemm.txt
./rocblas-bench -f gemm -r h --transposeA N --transposeB T -m 50176 -n 256 -k 128 --alpha 15360 --lda 50176 --ldb 256 --beta 0 --ldc 50176 >> gemm.txt
./rocblas-bench -f gemm -r h --transposeA N --transposeB T -m 50176 -n 256 -k 512 --alpha 15360 --lda 50176 --ldb 256 --beta 0 --ldc 50176 >> gemm.txt
./rocblas-bench -f gemm -r h --transposeA N --transposeB T -m 784 -n 1152 -k 128 --alpha 15360 --lda 784 --ldb 1152 --beta 0 --ldc 784 >> gemm.txt
./rocblas-bench -f gemm_strided_batched -r h --transposeA N --transposeB T -m 196 -n 1024 -k 256 --alpha 15360 --lda 196 --stride_a 50176 --ldb 1024 --stride_b 0 --beta 0 --ldc 196 --stride_c 200704 --batch 64 >> gemm.sb.txt
./rocblas-bench -f gemm_strided_batched -r h --transposeA N --transposeB T -m 196 -n 256 -k 1024 --alpha 15360 --lda 196 --stride_a 200704 --ldb 256 --stride_b 0 --beta 0 --ldc 196 --stride_c 50176 --batch 64 >> gemm.sb.txt
./rocblas-bench -f gemm_strided_batched -r h --transposeA N --transposeB T -m 3025 -n 256 -k 64 --alpha 15360 --lda 3025 --stride_a 193600 --ldb 256 --stride_b 0 --beta 0 --ldc 3025 --stride_c 774400 --batch 64 >> gemm.sb.txt
./rocblas-bench -f gemm_strided_batched -r h --transposeA N --transposeB T -m 3025 -n 64 -k 256 --alpha 15360 --lda 3025 --stride_a 774400 --ldb 64 --stride_b 0 --beta 0 --ldc 3025 --stride_c 193600 --batch 64 >> gemm.sb.txt
./rocblas-bench -f gemm_strided_batched -r h --transposeA N --transposeB T -m 3025 -n 64 -k 64 --alpha 15360 --lda 3025 --stride_a 193600 --ldb 64 --stride_b 0 --beta 0 --ldc 3025 --stride_c 193600 --batch 64 >> gemm.sb.txt
./rocblas-bench -f gemm_strided_batched -r h --transposeA N --transposeB T -m 3136 -n 256 -k 64 --alpha 15360 --lda 3136 --stride_a 200704 --ldb 256 --stride_b 0 --beta 0 --ldc 3136 --stride_c 802816 --batch 64 >> gemm.sb.txt
./rocblas-bench -f gemm_strided_batched -r h --transposeA N --transposeB T -m 3136 -n 64 -k 256 --alpha 15360 --lda 3136 --stride_a 802816 --ldb 64 --stride_b 0 --beta 0 --ldc 3136 --stride_c 200704 --batch 64 >> gemm.sb.txt
./rocblas-bench -f gemm_strided_batched -r h --transposeA N --transposeB T -m 3136 -n 64 -k 64 --alpha 15360 --lda 3136 --stride_a 200704 --ldb 64 --stride_b 0 --beta 0 --ldc 3136 --stride_c 200704 --batch 64 >> gemm.sb.txt
./rocblas-bench -f gemm_strided_batched -r h --transposeA N --transposeB T -m 49 -n 2048 -k 512 --alpha 15360 --lda 49 --stride_a 25088 --ldb 2048 --stride_b 0 --beta 0 --ldc 49 --stride_c 100352 --batch 64 >> gemm.sb.txt
./rocblas-bench -f gemm_strided_batched -r h --transposeA N --transposeB T -m 49 -n 512 -k 2048 --alpha 15360 --lda 49 --stride_a 100352 --ldb 512 --stride_b 0 --beta 0 --ldc 49 --stride_c 25088 --batch 64 >> gemm.sb.txt
./rocblas-bench -f gemm_strided_batched -r h --transposeA N --transposeB T -m 784 -n 128 -k 512 --alpha 15360 --lda 784 --stride_a 401408 --ldb 128 --stride_b 0 --beta 0 --ldc 784 --stride_c 100352 --batch 64 >> gemm.sb.txt
./rocblas-bench -f gemm_strided_batched -r h --transposeA N --transposeB T -m 784 -n 512 -k 128 --alpha 15360 --lda 784 --stride_a 100352 --ldb 512 --stride_b 0 --beta 0 --ldc 784 --stride_c 401408 --batch 64 >> gemm.sb.txt
