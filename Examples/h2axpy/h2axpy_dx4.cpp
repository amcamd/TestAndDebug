#include<iostream>
#include<hip/hip_runtime.h>
#include<hip/hip_runtime_api.h>
#include "rocblas.h"

typedef __fp16 half8 __attribute__((ext_vector_type(8)));
typedef __fp16 half2 __attribute__((ext_vector_type(2)));

extern "C" half2 __v_pk_fma_f16(half2, half2, half2) __asm("llvm.fma.v2f16");

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
void haxpy_half8_mod(int n, const __fp16 alpha, const __fp16 *x, __fp16 *y)
{
    int tid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    int index = ((n / 8) * 8) + tid;

    if (index < n) y[index] = alpha * x[index] + y[index];
}

__global__ void 
haxpy_half8(int n8, half2 alpha, const __fp16 *xx_fp16, __fp16 *yy_fp16) 
{
    int tid = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;

    half8 *xx = (half8 *)(xx_fp16);
    half8 *yy = (half8 *)(yy_fp16);

    half2 y0, y1, y2, y3;
    half2 x0, x1, x2, x3;
    half2 z0, z1, z2, z3;

    if(tid*8 < n8) 
    {
        y0[0] = yy[tid][0];
        y0[1] = yy[tid][1];
        y1[0] = yy[tid][2];
        y1[1] = yy[tid][3];
        y2[0] = yy[tid][4];
        y2[1] = yy[tid][5];
        y3[0] = yy[tid][6];
        y3[1] = yy[tid][7];

        x0[0] = xx[tid][0];
        x0[1] = xx[tid][1];
        x1[0] = xx[tid][2];
        x1[1] = xx[tid][3];
        x2[0] = xx[tid][4];
        x2[1] = xx[tid][5];
        x3[0] = xx[tid][6];
        x3[1] = xx[tid][7];

        z0 = __v_pk_fma_f16(alpha, x0, y0);
        z1 = __v_pk_fma_f16(alpha, x1, y1);
        z2 = __v_pk_fma_f16(alpha, x2, y2);
        z3 = __v_pk_fma_f16(alpha, x3, y3);

        yy[tid][0] = z0[0];
        yy[tid][1] = z0[1];
        yy[tid][2] = z1[0];
        yy[tid][3] = z1[1];
        yy[tid][4] = z2[0];
        yy[tid][5] = z2[1];
        yy[tid][6] = z3[0];
        yy[tid][7] = z3[1];
    }
}

template<typename T>
__global__ void
axpy_kernel_host_scalar(hipLaunchParm lp,
    int n,
    const T alpha,
    const T *x, rocblas_int incx,
    T *y,  rocblas_int incy)
{
    int tid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    if(incx >= 0 && incy >= 0)
    {
        if ( tid < n )
        {
            y[tid*incy] +=  (alpha) * x[tid * incx];
        }
    }
    else if(incx < 0 && incy < 0)
    {
        if (tid < n)
        {
            y[(1 - n + tid) * incy] +=  (alpha) * x[(1 - n + tid) * incx];
        }
    }
    else if (incx >=0)
    {
        if (tid < n)
        {
            y[(1 - n + tid) * incy] +=  (alpha) * x[tid * incx];
        }
    }
    else
    {
        if (tid < n)
        {
            y[tid * incy] +=  (alpha) * x[(1 - n + tid) * incx];
        }
    }
}
  


extern "C"
rocblas_status
rocblas_haxpy(rocblas_handle handle,
    rocblas_int n,
    const __fp16 *alpha,
    const __fp16 *x, rocblas_int incx,
          __fp16 *y,  rocblas_int incy)
{
    if (nullptr == alpha)
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
    if ( n <= 0 ) return rocblas_status_success;

    if(1 != incx || 1 != incy)
    {
        std::cout << "1 != incx || 1 != incy" << std::endl;
        int blocks = (n-1)/ NB_X + 1;

        dim3 grid( blocks, 1, 1 );
        dim3 threads(NB_X, 1, 1);

        hipLaunchKernel(HIP_KERNEL_NAME(axpy_kernel_host_scalar),
            dim3(blocks), dim3(threads), 0, 0,
            n, *alpha, x, incx, y, incy);
    }
    else
    {

        rocblas_int n8 = (n/8) * 8;
        half2 half2_alpha;
        half2_alpha.x = *alpha;
        half2_alpha.y = *alpha;

        int blocks = (((n/8)-1) / NB) + 1;

        dim3 grid( blocks, 1, 1 );
        dim3 threads(NB, 1, 1);

        hipLaunchKernelGGL(haxpy_half8, dim3(grid), dim3(threads), 0, 0, 
            n8, half2_alpha, x, y);

        int mod_threads = n - n8;

        if (0 != mod_threads)
        {
            hipLaunchKernelGGL(haxpy_half8_mod, dim3(1, 1, 1), dim3(mod_threads, 1, 1), 
                0, 0, n, *alpha, x, y);
        }
    }

    return rocblas_status_success;
}

int parse_args(int argc, char *argv[], int &n, int &incx, int &incy)
{
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
    __fp16 alpha = 3.0;
    if (parse_args(argc, argv, n, incx, incy))
    {
        usage(argv);
        return -1;
    }
    std::cout << "n, n%8 = " << n << ", " << n%8;
    std::cout << "      NB, n/NB, n%NB = " << NB << ", " << n/NB << ", " << n%NB;
    std::cout << "      incx, incy = " << incx << ", " << incy << std::endl;
        
    std::vector<__fp16> X(n*incx), Y(n*incy), Ycopy(n*incy);

    for(int i = 0; i < X.size(); i++)
    { 
        X[i] = __fp16((i+4)%64);
    }
    for(int i = 0; i < Y.size(); i++)
    { 
        Y[i] = __fp16(i%128); 
    }

    Ycopy = Y;
    int sizeX = n * incx * sizeof(__fp16);
    int sizeY = n * incy * sizeof(__fp16);
    __fp16 *Xd, *Yd;
    CHECK_HIP_ERROR(hipMalloc(&Xd, sizeX));
    CHECK_HIP_ERROR(hipMalloc(&Yd, sizeY));
    CHECK_HIP_ERROR(hipMemcpy(Xd, X.data(), sizeX, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(Yd, Y.data(), sizeY, hipMemcpyHostToDevice));

    rocblas_handle handle;
    CHECK_ROCBLAS_ERROR(rocblas_create_handle(&handle));

    hipSetDevice(1);

    auto start = std::chrono::high_resolution_clock::now();

    CHECK_ROCBLAS_ERROR(rocblas_haxpy( handle, n, &alpha, Xd, incx, Yd, incy));

    CHECK_HIP_ERROR(hipDeviceSynchronize());

    auto stop = std::chrono::high_resolution_clock::now();

    CHECK_HIP_ERROR(hipMemcpy(Y.data(), Yd, sizeY, hipMemcpyDeviceToHost));

    int even_error = 0;
    for(int i = 0; i < n; i+=2) {
        float out = float(alpha) * float(X[i*incx]) + float(Ycopy[i*incy]);
        if(float(Y[i*incy]) != out) {
            if(even_error < 20) 
                std::cerr<<"Bad even output: "<<float(Y[i*incy])<<" at: "<<i<<" Expected: "<<out<<std::endl;
                std::cerr<<"alpha, X[i], Ycopy[i] = "<<float(alpha)<<", "<<
                    float(X[i*incx])<<", "<<float(Ycopy[i*incy])<<std::endl;
            even_error++;
        }
    }
    if(0 == even_error)std::cout<<"---All even pass---    ";

    int odd_error = 0;
    for(int i = 1; i < n; i+=2) {
        float out = float(alpha) * float(X[i*incx]) + float(Ycopy[i*incy]);
        if(float(Y[i*incy]) != out) {
            if(odd_error < 20) 
                std::cerr<<"Bad odd output: "<<float(Y[i*incy])<<" at: "<<i<<" Expected: "<<out<<std::endl;
            odd_error++;
        }
    }
    if(0 == odd_error) std::cout<<"---All odd pass---   ";

    double elapsedSec = std::chrono::duration_cast<std::chrono::duration<double>>(stop - start).count();
    double perf = (double)(n*2)/1.0E12/elapsedSec;
    std::cout<<perf<<" TFLOPs"<<std::endl;

    CHECK_HIP_ERROR(hipFree(Xd));
    CHECK_HIP_ERROR(hipFree(Yd));
}
