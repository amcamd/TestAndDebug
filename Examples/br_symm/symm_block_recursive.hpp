
template<typename T>
rocblas_status symm_block_recursive( bool verbose, rocblas_side side, rocblas_fill uplo,
      rocblas_int m, rocblas_int n, T alpha,
      T *a, rocblas_int lda,
      T *b, rocblas_int ldb, T beta,
      T* c, rocblas_int ldc);

template<typename T>
rocblas_status symm_strided_block_recursive( bool verbose, rocblas_side side, rocblas_fill uplo,
      rocblas_int m, rocblas_int n, T alpha,
      T *a, rocblas_int lda,
      T *b, rocblas_int ldb, T beta,
      T *c, rocblas_int ldc);
