
I am testing trsm with     - { M: 4, N: 47000, lda: 47000, ldb: 4 }

internally it calls:


rocblas_internal_gemm_template(rocblas_handle    handle,
                                   rocblas_operation trans_a,
                                   rocblas_operation trans_b,
                                   rocblas_int       m,
                                   rocblas_int       n,
                                   rocblas_int       k,
                                   const TScal*      alpha,
                                   const TConstPtr*  A,
                                   rocblas_stride    offset_a,
                                   rocblas_int       lda,
                                   rocblas_stride    stride_a,
                                   const TConstPtr*  B,
                                   rocblas_stride    offset_b,
                                   rocblas_int       ldb,
                                   rocblas_stride    stride_b,
                                   const TScal*      beta,
                                   TPtr*             C,
                                   rocblas_stride    offset_c,
                                   rocblas_int       ldc,
                                   rocblas_stride    stride_c,
                                   rocblas_int       batch_count)

trans_a, trans_b = N, N
m, n, k, batch_count = 32, 32, 32, 367
offset_a, lda, stride_a = 0, 47000, 6016128
offset_b, ldb, stride_b = 0, 128, 16384
offset_c, ldc, stride_c = 0, 64, 4096


It fails with:

Memory access fault by GPU node-1 (Agent handle: 0x2fee6e0) on address 0x7f0b54025000. Reason: Page not present or supervisor privilege.


My best guess is that the problem is batch_count * stride_a = 367 * 6016128 = 2207918976 > 2147483647
