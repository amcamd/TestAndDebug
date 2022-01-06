
template<typename T> rocblas_status symm_reference( rocblas_side      side,
                                                    rocblas_fill      uplo,
                                                    rocblas_int       m,
                                                    rocblas_int       n,
                                                    T                 alpha,
                                                    T                 *a,
                                                    rocblas_int       lda,
                                                    T                 *b,
                                                    rocblas_int       ldb,
                                                    T                 beta,
                                                    T                 *c,
                                                    rocblas_int       ldc);

template<typename T>
rocblas_status symm_st_bat_ref(
     rocblas_side      side,
     rocblas_fill      uplo,
     rocblas_int       m,
     rocblas_int       n,
     T                 alpha,
     T                 *a,
     rocblas_int       lda, rocblas_stride stride_a,
     T                 *b,
     rocblas_int       ldb, rocblas_stride stride_b,
     T                 beta,
     T                 *c,
     rocblas_int       ldc, rocblas_stride stride_c,
     rocblas_int       batch_count);
