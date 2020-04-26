#include "rocblas.h"

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

template<bool right_side, typename TConstPtr, typename TPtr>
__global__ void dgmm_kernel(
                           rocblas_int m,
                           rocblas_int n,
                           TConstPtr A,
                           rocblas_int offsetA,
                           rocblas_int lda,
                           rocblas_stride strideA,
                           TConstPtr x,
                           rocblas_int offset_x,
                           rocblas_int incx,
                           rocblas_stride stridex,
                           TPtr C,
                           rocblas_int offsetC,
                           rocblas_int ldc,
                           rocblas_stride strideC)
{
    ptrdiff_t tx = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    ptrdiff_t ty = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;

    if(tx < m && ty < n)
    {
if(right_side)
{
        C[tx + ldc * ty] = A[tx + lda * ty]* x[offset_x + ty * incx];
}
else
{
        C[tx + ldc * ty] = A[tx + lda * ty]* x[offset_x + tx * incx];
}
    }
}


template<typename T> 
rocblas_status rocblas_dgmm_template( 
        rocblas_handle handle,
        rocblas_side side, 
        rocblas_int m, rocblas_int n,
        T *a, rocblas_int lda,
        T *x, rocblas_int incx,
        T *c, rocblas_int ldc)
{
    if (m < 0 || n < 0 || lda < m || ldc < m)
    {
        return rocblas_status_invalid_size;
    }
    if (m == 0 || n == 0) return rocblas_status_success;

    if(!a || !c )
    {
        return rocblas_status_invalid_pointer;
    }

    // in case of negative incx shift pointer to end of data for negative indexing
    ptrdiff_t offset_x = (incx < 0) ? - ptrdiff_t(incx) * (n - 1) : 0;

    hipStream_t rocblas_stream;
    CHECK_HIP_ERROR(hipStreamCreate(&rocblas_stream));

    rocblas_int batch_count = 1;
    ptrdiff_t offsetA = 0, offsetC = 0;
    rocblas_stride strideA = n * lda;
    rocblas_stride strideC = n * ldc;
    rocblas_stride stridex = rocblas_side_right == side ? n : m ;

    static constexpr int GEMV_DIM_X = 128;
    static constexpr int GEMV_DIM_Y = 8;
    rocblas_int          blocksX    = (m - 1) / GEMV_DIM_X + 1;
    rocblas_int          blocksY    = (n - 1) / GEMV_DIM_Y + 1;

    dim3 grid(blocksX, blocksY, batch_count);
    dim3 threads(GEMV_DIM_X, GEMV_DIM_Y);

    if (side == rocblas_side_left)
    {
         hipLaunchKernelGGL(dgmm_kernel<false>,
                           grid,
                           threads,
                           0,
                           rocblas_stream,
                           m,
                           n,
                           a,
                           offsetA,
                           lda,
                           strideA,
                           x,
                           offset_x,
                           incx,
                           stridex,
                           c,
                           offsetC,
                           ldc,
                           strideC);
    }
    else
    {
        hipLaunchKernelGGL(dgmm_kernel<true>,
                           grid,
                           threads,
                           0,
                           rocblas_stream,
                           m,
                           n,
                           a,
                           offsetA,
                           lda,
                           strideA,
                           x,
                           offset_x,
                           incx,
                           stridex,
                           c,
                           offsetC,
                           ldc,
                           strideC);
    }
    return rocblas_status_success;
}

template<typename T> 
rocblas_status rocblas_dgmm( 
        rocblas_side side, 
        rocblas_int m, rocblas_int n,
        T *ha, rocblas_int lda,
        T *hx, rocblas_int incx,
        T *hc, rocblas_int ldc)
{
    rocblas_status status = rocblas_status_success;

    rocblas_int size_a = lda * n;
    rocblas_int size_c = ldc * n;
    rocblas_int incx_pos = incx > 0 ? incx : -incx;
    rocblas_int size_x = incx_pos * (rocblas_side_right == side ? n : m);

    T *da, *dc, *dx;
    CHECK_HIP_ERROR(hipMalloc(&da, size_a * sizeof(T)));
    CHECK_HIP_ERROR(hipMalloc(&dc, size_c * sizeof(T)));
    CHECK_HIP_ERROR(hipMalloc(&dx, size_x * sizeof(T)));

    CHECK_HIP_ERROR( hipMemcpy(da, ha, sizeof(T) * size_a, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR( hipMemcpy(dc, hc, sizeof(T) * size_c, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR( hipMemcpy(dx, hx, sizeof(T) * size_x, hipMemcpyHostToDevice));

    rocblas_handle handle;
    CHECK_ROCBLAS_ERROR(rocblas_create_handle(&handle));

    status = rocblas_dgmm_template(handle, side, m, n, da, lda, dx, incx, dc, ldc);

    CHECK_HIP_ERROR( hipMemcpy(hc, dc, sizeof(T) * size_c, hipMemcpyDeviceToHost));

    return status;
}
