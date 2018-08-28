#include<iostream>
#include<hip/hip_runtime.h>
#include<hip/hip_runtime_api.h>
#include "rocblas.h"

typedef _Float16 half8 __attribute__((ext_vector_type(8)));
typedef _Float16 half2 __attribute__((ext_vector_type(2)));

extern "C" __device__ half2 __v_pk_fma_f16(half2, half2, half2) __asm("llvm.fma.v2f16");

#define NB 128
#define NB_X 256

#define CHECK_HIP_ERROR(error) \
if (error != hipSuccess) { \
    fprintf(stderr, "error: '%s'(%d) at %s:%d\n", hipGetErrorString(error), error,__FILE__, __LINE__); \
    exit(EXIT_FAILURE);\
}

#define CHECK_ROCBLAS_ERROR(error) \
if (error != rocblas_status_success) { \
    fprintf(stderr, "rocBLAS ERROR: "); \
    if(error == rocblas_status_invalid_handle)fprintf(stderr, "rocblas_status_invalid_handle"); \
    if(error == rocblas_status_not_implemented)fprintf(stderr, " rocblas_status_not_implemented"); \
    if(error == rocblas_status_invalid_pointer)fprintf(stderr, "rocblas_status_invalid_pointer"); \
    if(error == rocblas_status_invalid_size)fprintf(stderr, "rocblas_status_invalid_size"); \
    if(error == rocblas_status_memory_error)fprintf(stderr, "rocblas_status_memory_error"); \
    if(error == rocblas_status_internal_error)fprintf(stderr, "rocblas_status_internal_error"); \
    fprintf(stderr, "\n"); \
    exit(EXIT_FAILURE); \
}

//__global__
//void haxpy_half8_mod(int n, const _Float16 alpha, const _Float16 *x, _Float16 *y)
//{
//    int tid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
//
//    int index = ((n / 8) * 8) + tid;
//
//    if (index < n) y[index] = alpha * x[index] + y[index];
//}
//
//__global__ void 
//haxpy_half8(int n8, half2 alpha, const _Float16 *xx_fp16, _Float16 *yy_fp16) 
//{
//    int tid = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;
//
//    half8 *xx = (half8 *)(xx_fp16);
//    half8 *yy = (half8 *)(yy_fp16);
//
//    half2 y0, y1, y2, y3;
//    half2 x0, x1, x2, x3;
//    half2 z0, z1, z2, z3;
//
//    if(tid*8 < n8) 
//    {
//        y0[0] = yy[tid][0];
//        y0[1] = yy[tid][1];
//        y1[0] = yy[tid][2];
//        y1[1] = yy[tid][3];
//        y2[0] = yy[tid][4];
//        y2[1] = yy[tid][5];
//        y3[0] = yy[tid][6];
//        y3[1] = yy[tid][7];
//
//        x0[0] = xx[tid][0];
//        x0[1] = xx[tid][1];
//        x1[0] = xx[tid][2];
//        x1[1] = xx[tid][3];
//        x2[0] = xx[tid][4];
//        x2[1] = xx[tid][5];
//        x3[0] = xx[tid][6];
//        x3[1] = xx[tid][7];
//
//        z0 = __v_pk_fma_f16(alpha, x0, y0);
//        z1 = __v_pk_fma_f16(alpha, x1, y1);
//        z2 = __v_pk_fma_f16(alpha, x2, y2);
//        z3 = __v_pk_fma_f16(alpha, x3, y3);
//
//        yy[tid][0] = z0[0];
//        yy[tid][1] = z0[1];
//        yy[tid][2] = z1[0];
//        yy[tid][3] = z1[1];
//        yy[tid][4] = z2[0];
//        yy[tid][5] = z2[1];
//        yy[tid][6] = z3[0];
//        yy[tid][7] = z3[1];
//    }
//}
//
//template<typename T>
//__global__ void
//axpy_kernel_host_scalar(hipLaunchParm lp,
//    int n,
//    const T alpha,
//    const T *x, rocblas_int incx,
//    T *y,  rocblas_int incy)
//{
//    int tid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
//    if(incx >= 0 && incy >= 0)
//    {
//        if ( tid < n )
//        {
//            y[tid*incy] +=  (alpha) * x[tid * incx];
//        }
//    }
//    else if(incx < 0 && incy < 0)
//    {
//        if (tid < n)
//        {
//            y[(1 - n + tid) * incy] +=  (alpha) * x[(1 - n + tid) * incx];
//        }
//    }
//    else if (incx >=0)
//    {
//        if (tid < n)
//        {
//            y[(1 - n + tid) * incy] +=  (alpha) * x[tid * incx];
//        }
//    }
//    else
//    {
//        if (tid < n)
//        {
//            y[tid * incy] +=  (alpha) * x[(1 - n + tid) * incx];
//        }
//    }
//}
//  

template<typename T>
__global__ void
gemv_kernel_host_scalar(hipLaunchParm lp,
        rocblas_int n1, 
        rocblas_int n2,
        const T *a,
        rocblas_int lda,
        const T *x,
        T *y)
{
    int tid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    if(tid < n1)
    {
        for(int i = 0; i < n2; i++)
        {
            y[tid] += a[tid + i*lda] * x[i];
        }
    }

}

//  CHECK_ROCBLAS_ERROR(test_hgemv( handle, n1, n2, Ad, lda, Xd, Yd));

extern "C"
rocblas_status
test_hgemv(rocblas_handle handle,
    rocblas_int n1, 
    rocblas_int n2,
    const _Float16 *a, 
    rocblas_int lda,
    const _Float16 *x,
          _Float16 *y)
{
    if (nullptr == a)
        return rocblas_status_invalid_pointer;
    else if (nullptr == x)
        return rocblas_status_invalid_pointer;
    else if (nullptr == y)
        return rocblas_status_invalid_pointer;
    else if(nullptr == handle)
        return rocblas_status_invalid_handle;

    /*
     * Quick return if possible. Not Argument error
     */
    if ( n1 <= 0 || n2 <= 0 ) return rocblas_status_success;

    int blocks = (n1-1)/ NB_X + 1;

    dim3 grid( blocks, 1, 1 );
    dim3 threads(NB_X, 1, 1);

    hipLaunchKernel(HIP_KERNEL_NAME(gemv_kernel_host_scalar),
            dim3(blocks), dim3(threads), 0, 0,
            n1, n2, a, lda, x, y);

    return rocblas_status_success;
}

int parse_args(int argc, char *argv[], int &n, int &incx, int &incy)
{
    // default values
    n = 4;
    incx = 1;
    incy = 1;

    while (argc > 1)
    {
        if (argv[1][0] == '-')
        {
            switch (argv[1][1])
            {
                case 'n':
                    n = atoi(&argv[1][2]);
                    break;
                case 'x':
                    incx = atoi(&argv[1][2]);
                    break;
                case 'y': 
                    incy= atoi(&argv[1][2]);
                    break;
                default:
                    printf("Wrong Argument: %s\n", argv[1]);
                    return (1);
            }
        }
        else
        {
            printf("Wrong Argument: %s\n", argv[1]);
            return (1);
        }
        ++argv;
        --argc;
    }
    return (0);
}

void usage(char *argv[])
{
    std::cout << "Usage: " << argv[0];
    std::cout << " -n<dimension> -x<incx> -y<incy>" << std::endl;
}

int main(int argc, char *argv[]) 
{
    int n=0, incx=1, incy=1;
    _Float16 alpha = 1.0;
    if (parse_args(argc, argv, n, incx, incy))
    {
        usage(argv);
        return -1;
    }
    std::cout << "n, incx, incy = " << n << ", " << incx << ", " << incy << std::endl;
        
    int n1 = n;
    int n2 = n;
    int lda = n1;
    std::vector<_Float16> A(lda*n2), X(n2), Y(n2), Y_gold(n2);
    int sizeX = X.size() * sizeof(_Float16);
    int sizeY = Y.size() * sizeof(_Float16);
    int sizeA = A.size() * sizeof(_Float16);

    for(int i1 = 0; i1 < n; i1++)
    {
        for(int i2 = 0; i2 < n; i2++)
        {
            A[i1+i2*lda] = static_cast<_Float16>(1);
        }
    }
    std::cout << "-----------A[" << n1 << "," << n2 << "]-------------------" << std::endl;
    for(int i1 = 0; i1 < n1; i1++)
    {
        for(int i2 = 0; i2 < n2-1; i2++){std::cout << static_cast<float>(A[i1+i2*lda]) << ",";}
        std::cout << static_cast<float>(A[i1+(n2-1)*lda]) << std::endl;
    }

    for(int i = 0; i < X.size(); i++)
    { 
        X[i] = static_cast<_Float16>(1);
    }
    for(int i = 0; i < Y.size(); i++)
    { 
        Y[i] = static_cast<_Float16>(1); 
    }

    Y_gold = Y;
    for(int i1 = 0; i1 < n; i1++)
    {
        _Float16 t = 0.0;
        for(int i2 = 0; i2 < n; i2++)
        {
            t += A[i1+i2*lda] * X[i2];
        }
        Y_gold[i1] += alpha * t;
    }

    std::cout << "--------------x,y,y_gold---------------------" << std::endl;
    for(int i = 0; i < X.size(); i++){std::cout << static_cast<float>(X[i]) << ",";}; std::cout << std::endl;
    for(int i = 0; i < Y.size(); i++){std::cout << static_cast<float>(Y[i]) << ",";}; std::cout << std::endl;
    for(int i = 0; i < Y_gold.size(); i++){std::cout << static_cast<float>(Y_gold[i]) << ",";}; std::cout << std::endl;
//  std::cout << "---------------------------------------------" << std::endl;

    _Float16 *Ad, *Xd, *Yd;
    CHECK_HIP_ERROR(hipMalloc(&Ad, sizeA));
    CHECK_HIP_ERROR(hipMalloc(&Xd, sizeX));
    CHECK_HIP_ERROR(hipMalloc(&Yd, sizeY));
    CHECK_HIP_ERROR(hipMemcpy(Ad, A.data(), sizeA, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(Xd, X.data(), sizeX, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(Yd, Y.data(), sizeY, hipMemcpyHostToDevice));

    rocblas_handle handle;
    CHECK_ROCBLAS_ERROR(rocblas_create_handle(&handle));

    hipSetDevice(1);

    CHECK_ROCBLAS_ERROR(test_hgemv( handle, n1, n2, Ad, lda, Xd, Yd));

    CHECK_HIP_ERROR(hipMemcpy(Y.data(), Yd, sizeY, hipMemcpyDeviceToHost));

    std::cout << "----------------y----------------------------" << std::endl;
    for(int i = 0; i < Y.size(); i++){std::cout << static_cast<float>(Y[i]) << ",";}; std::cout << std::endl;
    std::cout << "---------------------------------------------" << std::endl;


    _Float16 max_error = 0;
    for(int i = 0; i < n1; i++)
    {
        _Float16 error = (Y[i] - Y_gold[i]) / Y_gold[i];
        _Float16 abs_error = error >= 0 ? error : -error;
        max_error = max_error > abs_error ? max_error : abs_error;
//      std::cout << "error, abs_error, max_error = " << static_cast<float>(error) << ", " << static_cast<float>(abs_error) << ", " << static_cast<float>(max_error) << std::endl;
    }
    std::cout << "max_error = " << static_cast<float>(max_error) << std::endl;

    CHECK_HIP_ERROR(hipFree(Ad));
    CHECK_HIP_ERROR(hipFree(Xd));
    CHECK_HIP_ERROR(hipFree(Yd));
}
