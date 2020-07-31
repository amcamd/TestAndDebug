template <typename T>
void gemm_batched_solution(int m, int n, int k,
                    const T alpha, const T* const dA_array[], int lda,
                                   const T* const dB_array[], int ldb,
                    const T beta,        T* const dC_array[], int ldc,
                    int batch_count, hipStream_t stream);
