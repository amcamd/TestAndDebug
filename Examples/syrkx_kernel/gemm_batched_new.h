
    template <typename T, typename TConstPtr, typename TPtr>
    void gemm_batched_solution(rocblas_operation trans_a,
                               rocblas_operation trans_b,
                               rocblas_int       m,
                               rocblas_int       n,
                               rocblas_int       k,
                               const T           alpha,
                               TConstPtr*        dA_array,
                               rocblas_int       lda,
                               rocblas_stride    stride_a,
                               TConstPtr*        dB_array,
                               rocblas_int       ldb,
                               rocblas_stride    stride_b,
                               const T           beta,
                               TPtr*             dC_array,
                               rocblas_int       ldc,
                               rocblas_stride    stride_c,
                               rocblas_int       batch_count,
                               hipStream_t       stream);
