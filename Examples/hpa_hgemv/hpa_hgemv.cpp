#include<iostream>
#include<hip/hip_runtime.h>
#include<hip/hip_runtime_api.h>
#include "rocblas.h"

typedef __fp16 half8 __attribute__((__vector_size__(8*sizeof(__fp16))));
typedef __fp16 half2 __attribute__((__vector_size__(2*sizeof(__fp16))));

extern "C" __device__ float __builtin_amdgcn_fdot2(half2, half2, float);

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


__global__
void gemv_kernel_host_scalar(hipLaunchParm   lp,
                        rocblas_int          n1, 
                        rocblas_int          n2_div_8,
                        const half8          *aa,
                        rocblas_int          lda_div_8,
                        const half8          *xx,
                        __fp16               *yy,
                        float                alpha,
                        float                beta)
{
    int tid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    if(tid < n1)
    {
        float c = 0.0;
        for(int i8 = 0; i8 < n2_div_8; i8 += 1)
        {
            half2 a0, a1, a2, a3;
            half2 x0, x1, x2, x3;

            a0[0] = aa[i8 + tid*lda_div_8][0];
            a0[1] = aa[i8 + tid*lda_div_8][1];
            a1[0] = aa[i8 + tid*lda_div_8][2];
            a1[1] = aa[i8 + tid*lda_div_8][3];
            a2[0] = aa[i8 + tid*lda_div_8][4];
            a2[1] = aa[i8 + tid*lda_div_8][5];
            a3[0] = aa[i8 + tid*lda_div_8][6];
            a3[1] = aa[i8 + tid*lda_div_8][7];

            x0[0] = xx[i8][0];
            x0[1] = xx[i8][1];
            x1[0] = xx[i8][2];
            x1[1] = xx[i8][3];
            x2[0] = xx[i8][4];
            x2[1] = xx[i8][5];
            x3[0] = xx[i8][6];
            x3[1] = xx[i8][7];

            c = __builtin_amdgcn_fdot2(a0, x0, c);
            c = __builtin_amdgcn_fdot2(a1, x1, c);
            c = __builtin_amdgcn_fdot2(a2, x2, c);
            c = __builtin_amdgcn_fdot2(a3, x3, c);
        }
        yy[tid] = static_cast<__fp16>(alpha * c) + static_cast<__fp16>(beta * yy[tid]);
    }
}

extern "C"
rocblas_status
test_hgemv(rocblas_handle handle,
    rocblas_int n1, 
    rocblas_int n2,
    const __fp16 *a, 
    rocblas_int lda,
    const __fp16 *x,
          __fp16 *y,
    float   alpha,
    float   beta)
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
                                    n1, 
                                    n2/8, 
                                    reinterpret_cast<const half8*>(a), 
                                    lda/8, 
                                    reinterpret_cast<const half8*>(x), 
                                    y, 
                                    alpha, 
                                    beta);

    return rocblas_status_success;
}

int parse_args(int argc, char *argv[], int &n, int &incx, int &incy)
{
    // default values

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
    int n=16, incx=1, incy=1;
    float alpha = 1.0;
    float beta  = 2.0;
    if (parse_args(argc, argv, n, incx, incy))
    {
        usage(argv);
        return -1;
    }
    std::cout << "n, incx, incy, alpha, beta = " << n << ", " << incx << ", " << incy << ", " 
              << alpha << ", " << beta << ", " << std::endl;
        
    int n1 = n;
    int n2 = n;
    int lda = n1;
    std::vector<__fp16> A(lda*n2), X(n2), Y(n2), Y_gold(n2);
    int sizeX = X.size() * sizeof(__fp16);
    int sizeY = Y.size() * sizeof(__fp16);
    int sizeA = A.size() * sizeof(__fp16);

    for(int i1 = 0; i1 < n; i1++)
    {
        for(int i2 = 0; i2 < n; i2++)
        {
            A[i1+i2*lda] = static_cast<__fp16>(rand()%10);
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
        X[i] = static_cast<__fp16>(rand()%10);
    }
    for(int i = 0; i < Y.size(); i++)
    { 
        Y[i] = static_cast<__fp16>(rand()%10); 
    }

    Y_gold = Y;
    for(int i1 = 0; i1 < n; i1++)
    {
        float t = 0.0;
        for(int i2 = 0; i2 < n; i2++)
        {
            t += static_cast<float>(A[i1*lda+i2]) * static_cast<float>(X[i2]);
        }
        Y_gold[i1] = static_cast<__fp16>(beta * static_cast<float>(Y_gold[i1]) + alpha * t);
    }

    std::cout << "--------------x,y,y_gold---------------------" << std::endl;
    for(int i = 0; i < X.size(); i++){std::cout << static_cast<float>(X[i]) << ",";}; std::cout << std::endl;
    for(int i = 0; i < Y.size(); i++){std::cout << static_cast<float>(Y[i]) << ",";}; std::cout << std::endl;
    for(int i = 0; i < Y_gold.size(); i++){std::cout << static_cast<float>(Y_gold[i]) << ",";}; std::cout << std::endl;

    __fp16 *Ad, *Xd, *Yd;
    CHECK_HIP_ERROR(hipMalloc(&Ad, sizeA));
    CHECK_HIP_ERROR(hipMalloc(&Xd, sizeX));
    CHECK_HIP_ERROR(hipMalloc(&Yd, sizeY));
    CHECK_HIP_ERROR(hipMemcpy(Ad, A.data(), sizeA, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(Xd, X.data(), sizeX, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(Yd, Y.data(), sizeY, hipMemcpyHostToDevice));

    rocblas_handle handle;
    CHECK_ROCBLAS_ERROR(rocblas_create_handle(&handle));

    hipSetDevice(1);

    CHECK_ROCBLAS_ERROR(test_hgemv( handle, n1, n2, Ad, lda, Xd, Yd, alpha, beta));

    CHECK_HIP_ERROR(hipMemcpy(Y.data(), Yd, sizeY, hipMemcpyDeviceToHost));

    std::cout << "----------------y----------------------------" << std::endl;
    for(int i = 0; i < Y.size(); i++){std::cout << static_cast<float>(Y[i]) << ",";}; std::cout << std::endl;
    std::cout << "---------------------------------------------" << std::endl;


    __fp16 max_error = 0;
    for(int i = 0; i < n1; i++)
    {
        __fp16 error = (Y[i] - Y_gold[i]) / Y_gold[i];
        __fp16 abs_error = error >= 0 ? error : -error;
        max_error = max_error > abs_error ? max_error : abs_error;
//      std::cout << "error, abs_error, max_error = " << static_cast<float>(error) << ", " << static_cast<float>(abs_error) << ", " << static_cast<float>(max_error) << std::endl;
    }
    std::cout << "max_error = " << static_cast<float>(max_error) << std::endl;

    CHECK_HIP_ERROR(hipFree(Ad));
    CHECK_HIP_ERROR(hipFree(Xd));
    CHECK_HIP_ERROR(hipFree(Yd));
}
