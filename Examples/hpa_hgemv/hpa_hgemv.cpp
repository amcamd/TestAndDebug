#include<iostream>
#include<limits>
#include<hip/hip_runtime.h>
#include<hip/hip_runtime_api.h>
#include "rocblas.h"

typedef __fp16 half8 __attribute__((__vector_size__(8*sizeof(__fp16))));
typedef __fp16 half2 __attribute__((__vector_size__(2*sizeof(__fp16))));

extern "C" __device__ float __builtin_amdgcn_fdot2(half2, half2, float);

#define NB 128
#define NB_x_16 256

#define CHECK_HIP_ERROR(error) \
if (error != hipSuccess) { \
    fprintf(stderr, "error: '%s'(%d) at %s:%d\n", hipGetErrorString(error), error,__FILE__, __LINE__); \
    exit(EXIT_FAILURE);\
}

#define CHECK_ROCBLAS_ERROR(error) \
if (error != rocblas_status_success) { \
    fprintf(stderr, "rocBLa_16S ERROR: "); \
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
        yy[tid] = static_cast<__fp16>((alpha * c) + (beta * static_cast<float>(yy[tid])));
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
     * Quick return if possible. Not a_16rgument error
     */
    if ( n1 <= 0 || n2 <= 0 ) return rocblas_status_success;

    int blocks = (n1-1)/ NB_x_16 + 1;

    dim3 grid( blocks, 1, 1 );
    dim3 threads(NB_x_16, 1, 1);

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

int parse_args(int argc, char *argv[], int &n1, int &n2, int &incx, int &incy, int &lda)
{
    if(argc >= 2)
    {
        for(int i = 1; i < argc; ++i)
        {
            std::string arg = argv[i];

            if((arg.at(0) == '-') || ((arg.at(0) == '-') && (arg.at(1) == '-')))
            {
                if((arg == "-n1") && (i + 1 < argc))
                {
                    n1 = atoi(argv[++i]);
                }
                else if((arg == "-n2") && (i + 1 < argc))
                {
                    n2 = atoi(argv[++i]);
                }
                else if((arg == "-incx") && (i + 1 < argc))
                {
                    incx = atoi(argv[++i]);
                }
                else if((arg == "-incy") && (i + 1 < argc))
                {
                    incy = atoi(argv[++i]);
                }
                else if((arg == "--lda") && (i + 1 < argc))
                {
                    lda = atoi(argv[++i]);
                }
                else
                {
                    std::cerr << "error with " << arg << std::endl;
                    std::cerr << "do not recognize option" << std::endl << std::endl;
                    return EXIT_FAILURE;
                }
            }
            else
            {
                std::cerr << "error with " << arg << std::endl;
                std::cerr << "option must start with - or --" << std::endl << std::endl;
                return EXIT_FAILURE;
            }
        }
    }
    return EXIT_SUCCESS;
}

void usage(char *argv[])
{
    std::cout << "Usage: " << argv[0];
    std::cout << " --n1<a dimension 1> --n2<a dimention 2> --incx<incx> --incy<incy> --lda<leading dimension a>" << std::endl;
}

int main(int argc, char *argv[]) 
{
    int n1 = 16; int incx=1;
    int n2 = 16; int incy=1;
    float alpha = 1.0;
    float beta  = 2.0;
    int lda = n1;
    if (parse_args(argc, argv, n1, n2, incx, incy, lda))
    {
        usage(argv);
        return -1;
    }
    std::cout << "n1, n2, incx, incy, alpha, beta = " << n1 << ", " << n2 << ", " << incx << ", " << incy << ", " 
              << alpha << ", " << beta << ", " << std::endl;
        
    int size_x = n2 * incx;
    int size_y = n1 * incy;
    int size_a = lda * n2;
    std::vector<__fp16> a_16(size_a), x_16(size_x), y_16(size_y), y_16_gold(size_y);
    std::vector<float> a_32(size_a), x_32(size_x), y_32(size_y), y_32_gold(size_y);

    // initialize a, x, y
    for(int i1 = 0; i1 < n1; i1++)
    {
        for(int i2 = 0; i2 < n2; i2++)
        {
            a_32[i1+i2*lda] = rand()%10;
            a_16[i1+i2*lda] = static_cast<__fp16>(a_32[i1+i2*lda]);
        }
    }

    for(int i = 0; i < x_16.size(); i++)
    { 
        x_32[i] = rand()%10;
        x_16[i] = static_cast<__fp16>(x_32[i]);
    }
    for(int i = 0; i < y_16.size(); i++)
    { 
        y_32[i] = rand()%10; 
        y_16[i] = static_cast<__fp16>(y_32[i]); 
    }

    x_16[0] = 65503.0;
    x_16[n2-1] = x_16[0];
    x_32[0] = static_cast<float>(x_16[0]);
    x_32[n2-1] = static_cast<float>(x_16[n2-1]);

    for(int i1 = 0; i1 < n1; i1++)
    {
        a_16[0+(lda*i1)] = 4.0;
        a_16[(n2-1)+(lda*i1)] =-4.0;

        a_32[0+(lda*i1)] = static_cast<float>(a_16[0+(lda*i1)]);
        a_32[(n2-1)+(lda*i1)] = static_cast<float>(a_16[(n2-1)+(lda*i1)]);
    }

    // calculate gold result on cpu
    y_16_gold = y_16;
    y_32_gold = y_32;
    for(int i1 = 0; i1 < n1; i1++)
    {
        __fp16  t_16 = 0.0;
        float   t_32 = 0.0;
        for(int i2 = 0; i2 < n2; i2++)
        {
            t_16 += a_16[i2+i1*lda] * x_16[i2];
            t_32 += a_16[i2+i1*lda] * x_16[i2];
        }
        y_16_gold[i1] = (static_cast<__fp16>(beta) * y_16_gold[i1]) + (static_cast<__fp16>(alpha) * t_16);
        y_32_gold[i1] = (beta * y_32_gold[i1]) + (alpha * t_32);
    }

    // print a, x, y, y_gold
    std::cout << "-----------a_16[" << n1 << "," << n2 << "]-------------------" << std::endl;
    for(int i1 = 0; i1 < n1; i1++)
    {
        for(int i2 = 0; i2 < n2-1; i2++)
        {
            std::cout << static_cast<float>(a_16[i1+i2*lda]) << ",";
        }
        std::cout << static_cast<float>(a_16[i1+(n2-1)*lda]) << std::endl;
    }
    std::cout << "--------------x,y,y_16_gold,y_32_gold-----------" << std::endl;
    for(int i = 0; i < x_16.size(); i++){std::cout << static_cast<float>(x_16[i]) << ",";}; std::cout << std::endl;
    for(int i = 0; i < y_16.size(); i++){std::cout << static_cast<float>(y_16[i]) << ",";}; std::cout << std::endl;
    for(int i = 0; i < y_16_gold.size(); i++){std::cout << static_cast<float>(y_16_gold[i]) << ",";}; std::cout << std::endl;
    for(int i = 0; i < y_32_gold.size(); i++){std::cout << static_cast<float>(y_32_gold[i]) << ",";}; std::cout << std::endl;

    // allocate a, x, y on device, and copy to device
    __fp16 *a1_16_d, *x_16_d, *y_16_d;
    CHECK_HIP_ERROR(hipMalloc(&a1_16_d, size_a * sizeof(__fp16)));
    CHECK_HIP_ERROR(hipMalloc(&x_16_d, size_x * sizeof(__fp16)));
    CHECK_HIP_ERROR(hipMalloc(&y_16_d, size_y * sizeof(__fp16)));
    CHECK_HIP_ERROR(hipMemcpy(a1_16_d, a_16.data(), size_a * sizeof(__fp16), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(x_16_d, x_16.data(), size_x * sizeof(__fp16), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(y_16_d, y_16.data(), size_y * sizeof(__fp16), hipMemcpyHostToDevice));

    // calculate y on gpu using hpa
    rocblas_handle handle;
    CHECK_ROCBLAS_ERROR(rocblas_create_handle(&handle));

    hipSetDevice(1);

    CHECK_ROCBLAS_ERROR(test_hgemv( handle, n1, n2, a1_16_d, lda, x_16_d, y_16_d, alpha, beta));

    CHECK_HIP_ERROR(hipMemcpy(y_16.data(), y_16_d, size_y * sizeof(__fp16), hipMemcpyDeviceToHost));

    std::cout << "----------------y----------------------------" << std::endl;
    for(int i = 0; i < y_16.size(); i++){std::cout << static_cast<float>(y_16[i]) << ",";}; std::cout << std::endl;
    std::cout << "---------------------------------------------" << std::endl;


    __fp16 max_error = 0;
    for(int i = 0; i < n1; i++)
    {
        __fp16 y_32_to_16_gold = static_cast<__fp16>(y_32_gold[i]);
        __fp16 error = (y_16[i] - y_32_to_16_gold) / y_32_to_16_gold;
        __fp16 abs_error = error >= 0 ? error : -error;
        max_error = max_error > abs_error ? max_error : abs_error;
//      std::cout << "error, abs_error, max_error = " << static_cast<float>(error) << ", " << static_cast<float>(abs_error) << ", " << static_cast<float>(max_error) << std::endl;
    }
    std::cout << "max_error = " << static_cast<float>(max_error) << std::endl;

    CHECK_HIP_ERROR(hipFree(a1_16_d));
    CHECK_HIP_ERROR(hipFree(x_16_d));
    CHECK_HIP_ERROR(hipFree(y_16_d));
}
