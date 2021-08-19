#include <vector>
#include <iostream>
#include <hip/hip_runtime.h>
#include <rocblas.h>

#ifndef CHECK_HIP_ERROR
#define CHECK_HIP_ERROR(error)                    \
    if(error != hipSuccess)                       \
    {                                             \
        fprintf(stderr,                           \
                "Hip error: '%s'(%d) at %s:%d\n", \
                hipGetErrorString(error),         \
                error,                            \
                __FILE__,                         \
                __LINE__);                        \
        exit(EXIT_FAILURE);                       \
    }
#endif

#ifndef CHECK_ROCBLAS_ERROR
#define CHECK_ROCBLAS_ERROR(error)                              \
    if(error != rocblas_status_success)                         \
    {                                                           \
        fprintf(stderr, "rocBLAS error: ");                     \
        if(error == rocblas_status_invalid_handle)              \
            fprintf(stderr, "rocblas_status_invalid_handle");   \
        if(error == rocblas_status_not_implemented)             \
            fprintf(stderr, " rocblas_status_not_implemented"); \
        if(error == rocblas_status_invalid_pointer)             \
            fprintf(stderr, "rocblas_status_invalid_pointer");  \
        if(error == rocblas_status_invalid_size)                \
            fprintf(stderr, "rocblas_status_invalid_size");     \
        if(error == rocblas_status_memory_error)                \
            fprintf(stderr, "rocblas_status_memory_error");     \
        if(error == rocblas_status_internal_error)              \
            fprintf(stderr, "rocblas_status_internal_error");   \
        fprintf(stderr, "\n");                                  \
        exit(EXIT_FAILURE);                                     \
    }
#endif

int main()
{
    int m = 64;
    int n = 512;
    int lda = m;

    std::vector<float> hA;
    float *dA;
    rocblas_handle handle;

    rocblas_create_handle(&handle);

    int a_size = n * lda;

    hA.resize(a_size);
    for (int i = 0; i < a_size; i++)
    {
        hA[i] = ((float)rand() / (float)RAND_MAX);
    }

    CHECK_HIP_ERROR(hipMalloc(&dA, a_size * sizeof(float)));
    CHECK_ROCBLAS_ERROR(rocblas_set_matrix(m, n, sizeof(float), hA.data(), lda, dA, lda));

    CHECK_HIP_ERROR(hipFree(dA));

    std::cout << "before return" << std::endl;
    return 0;
}
