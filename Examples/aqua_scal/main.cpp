#include <iostream>
#include <cstdio>
#include <iomanip>
#include <vector>
#include <random>
#include <limits>
#include <cstring>
#include <stdlib.h>
#include <chrono>
#include <hip/hip_runtime.h>
#include "rocblas.h"

#define NB 256

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

static void show_usage(char* argv[])
{
        std::cerr << "Usage: " << argv[0] << " <options>\n"
                  << "options:\n"
                  << "\t-h, --help\t\t\t\tShow this help message\n"
                  << "\t-v, --verbose\t\t\t\tverbose output\n"
                  << "\t-p \t\t\tp\t\tprecision s, d, c, z, h\n"
                  << "\t-n \t\t\tn\t\trocblas_gemm_ex argument n\n"
                  << "\t-i \t\t\ti\t\tnumber of hot cache calls\n"
                  << "\t-j \t\t\tj\t\tnumber of cold cache calls\n"
                  << std::endl;
}

static int parse_arguments(
        int argc,
        char *argv[],
        rocblas_int &n, 
        int64_t &n_hot,
        int64_t &n_cold,
        bool &verbose,
        char &precision)
{
    if(argc >= 2)
    {
        for(int i = 1; i < argc; ++i)
        {
            std::string arg = argv[i];
            if((arg.at(0) == '-') || ((arg.at(0) == '-') && (arg.at(1) == '-')))
            {
                if((arg == "-h") || (arg == "--help"))
                {
                    return EXIT_FAILURE;
                }
                if((arg == "-v") || (arg == "--verbose"))
                {
                    verbose = true;
                }
                else if((arg == "-p") && (i + 1 < argc))
                {
                    precision = *(argv[++i]);
                }
                else if((arg == "-n") && (i + 1 < argc))
                {
                    n = atoi(argv[++i]);
                }
                else if((arg == "-i") && (i + 1 < argc))
                {
                    n_hot = atoi(argv[++i]);
                }
                else if((arg == "-j") && (i + 1 < argc))
                {
                    n_cold = atoi(argv[++i]);
                }
            }
        }
    }
    return EXIT_SUCCESS;
}

/*
template <typename T>
void print_matrix(
            const char* name, std::vector<T>& A, rocblas_int m, rocblas_int n, rocblas_int lda)
{
    printf("---------- %s ----------\n", name);
    int max_i = 22;
    int max_j = 12;
    for(int i = 0; i < m && i < max_i; i++)
    {
        for(int j = 0; j < n && j < max_j; j++)
        {
            std::cout << std::setw(4) << float(A[i + j * lda]) << " ";
        }
        std::cout << "\n";
    }
}
*/

template <typename T>
void initialize_arrays(rocblas_int n, rocblas_int incx, T* x, T* x_ref)
{
    for (int i = 0; i < n; i++)
    {
        x[i*incx] = static_cast<T>(1.0);
        x_ref[i*incx] = x[i*incx];
    }
}

//template <typename T, typename = typename std::enable_if_t<std::is_same_v<T, float> ||
template <typename T, std::enable_if_t<std::is_same_v<T, float> ||
                                       std::is_same_v<T, double>>>

void initialize_arrays   (rocblas_int n, rocblas_int incx, T* x, T* x_ref)
//void initialize_arrays<T>(rocblas_int n, rocblas_int incx, T* x, T* x_ref)
{
    std::random_device rd;  // a seed source for the random number engine
    std::mt19937 generator(rd()); // mersenne_twister_engine seeded with rd()
    std::uniform_int_distribution<int> distribution(1, 6);

    for (int i = 0; i < n; i++)
    {
        x[i*incx] = static_cast<T>(distribution(generator));
        x_ref[i*incx] = x[i*incx];
    }
}

template <typename T>
bool ref_calc(rocblas_int n, rocblas_int incx, T alpha, T* hx_ref)
{
    for(int i = 0; i < n; i++)
    {
        hx_ref[i*incx] *= alpha;
    }
    return true;
}

template <typename T>
bool verify_solution(rocblas_int n, rocblas_int incx, T alpha, T* hx, T* hx_ref)
{
    for(int i = 0; i < n; i++)
    {
        if(hx[i*incx] != hx_ref[i*incx])
        {
            std::cout << "i, hx[i*incx], hx_ref[i*incx] = " << i << ", " << hx[i*incx] << ", " << hx_ref[i*incx] << std::endl;
            return false;
        }
    }
    return true;
}

template <typename T>
__global__ void Xscal(rocblas_int n, T alpha, T* x, rocblas_int incx)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    if(tid < n)
    {
        x[tid * int64_t(incx)] *= alpha;
    }
}


template <typename T>
__global__ void Xscal_float4(rocblas_int n, T alpha, T* x)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    if(tid < n)
    {
        x[tid] *= alpha;
    }
}

template <>
__global__ void Xscal_float4<float>(rocblas_int n, float alpha, float* x)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    float4 *x_float4_ptr = reinterpret_cast<float4 *>(x + tid*4);

    float4 x_float4 = *x_float4_ptr;

    x_float4.x *= alpha;
    x_float4.y *= alpha;
    x_float4.z *= alpha;
    x_float4.w *= alpha;

    *x_float4_ptr = x_float4;
}

template <typename T>
__global__ void Xscal_double2(rocblas_int n, T alpha, T* x)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    if(tid < n)
    {
        x[tid] *= alpha;
    }
}

template <>
__global__ void Xscal_double2<double>(rocblas_int n, double alpha, double* x)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    double2 *x_double2_ptr = reinterpret_cast<double2 *>(x + tid*2);

    double2 x_double2 = *x_double2_ptr;

    x_double2.x *= alpha;
    x_double2.y *= alpha;

    *x_double2_ptr = x_double2;
}

template <typename T>
void rocblas_Xscal( rocblas_handle handle, rocblas_int n, T alpha, T* dx, rocblas_int incx)
{
    std::cout << "need to specialize rocblas_Xscal templated function" << std::endl;
}
template <>
void rocblas_Xscal<float>( rocblas_handle handle, rocblas_int n, float alpha, float* dx, rocblas_int incx)
{
//  rocblas_handle handle;
//  rocblas_create_handle(&handle);

    rocblas_sscal(handle, n, &alpha, dx, incx);
}
template <>
void rocblas_Xscal<double>( rocblas_handle handle, rocblas_int n, double alpha, double* dx, rocblas_int incx)
{
//  rocblas_handle handle;
//  rocblas_create_handle(&handle);

    rocblas_dscal(handle, n, &alpha, dx, incx);
}
template <>
void rocblas_Xscal<rocblas_float_complex>( rocblas_handle handle, rocblas_int n, rocblas_float_complex alpha, rocblas_float_complex* dx, rocblas_int incx)
{
//  rocblas_handle handle;
//  rocblas_create_handle(&handle);

    rocblas_cscal(handle, n, &alpha, dx, incx);
}
template <>
void rocblas_Xscal<rocblas_double_complex>( rocblas_handle handle, rocblas_int n, rocblas_double_complex alpha, rocblas_double_complex* dx, rocblas_int incx)
{
//  rocblas_handle handle;
//  rocblas_create_handle(&handle);

    rocblas_zscal(handle, n, &alpha, dx, incx);
}

template <typename T>
void template_scal(rocblas_int n, rocblas_int incx, int64_t n_hot, int64_t n_cold)
{
    double ops = (std::is_same_v<float, T> || std::is_same_v<double, T>) ? n : 6*n;
    double gflops_rocblas, gflops_simple, gflops_mult2, gflops_mult4;
    int64_t n_launch = 500;

    rocblas_handle handle;
    rocblas_create_handle(&handle);

    std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
    std::chrono::duration<double> dur;
    double seconds = 0.0;

    T alpha = 10.0;
    T *dx, *hx, *hx_ref, *hx_calc;

    hx      = (T*)malloc(n*sizeof(T));
    hx_ref  = (T*)malloc(n*sizeof(T));
    hx_calc = (T*)malloc(n*sizeof(T));

    initialize_arrays(n, incx, hx, hx_ref);
    ref_calc(n, incx, alpha, hx_ref);

    CHECK_HIP_ERROR(hipMalloc(&dx, n * sizeof(T)));

    CHECK_HIP_ERROR(hipMemcpy(dx, hx, n * sizeof(T), hipMemcpyHostToDevice));

//  ------ rocBLAS -----
//  correctness
    CHECK_HIP_ERROR(hipMemcpy( dx, hx, n * sizeof(T), hipMemcpyHostToDevice));
        rocblas_Xscal( handle, n, alpha, dx, 1);
    CHECK_HIP_ERROR(hipMemcpy(hx_calc, dx, n * sizeof(T), hipMemcpyDeviceToHost));

    if (!verify_solution(n, incx, alpha, hx_calc, hx_ref)) std::cout << "rocBLAS; FAIL ";

//  timing
    for (int i = 0; i < n_cold; i++) { rocblas_Xscal( handle, n, alpha, dx, 1); };
    CHECK_HIP_ERROR(hipDeviceSynchronize());
    start = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < n_hot; i++)
        {
            rocblas_Xscal( handle, n, alpha, dx, 1);
        }

    CHECK_HIP_ERROR(hipDeviceSynchronize());
    end = std::chrono::high_resolution_clock::now(); dur= end - start; seconds = dur.count(); 
    gflops_rocblas = (ops * n_hot) / seconds / 1e9;

//  std::cout << "sec, gflops = " << seconds << ", " << gflops << std::endl; 

//  ----- simple -----
//  correctness
    int blocks = (n - 1) / NB + 1;
    dim3 grid(blocks, 1, 1);
    dim3 threads(NB, 1, 1);

    CHECK_HIP_ERROR(hipMemcpy( dx, hx, n * sizeof(T), hipMemcpyHostToDevice));
        hipLaunchKernelGGL(Xscal, grid, threads, 0, 0, n, alpha, dx, 1);
    CHECK_HIP_ERROR(hipMemcpy(hx_calc, dx, n * sizeof(T), hipMemcpyDeviceToHost));

    if (!verify_solution(n, incx, alpha, hx_calc, hx_ref)) std::cout << " simple; FAIL ";

//  timing
    for (int i = 0; i < n_cold; i++) { hipLaunchKernelGGL(Xscal, grid, threads, 0, 0, n, alpha, dx, 1); };
    CHECK_HIP_ERROR(hipDeviceSynchronize());
    start = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < n_hot; i++)
        {
            hipLaunchKernelGGL(Xscal, grid, threads, 0, 0, n, alpha, dx, 1);
        }

    CHECK_HIP_ERROR(hipDeviceSynchronize());
    end = std::chrono::high_resolution_clock::now(); dur= end - start; seconds = dur.count(); 
    gflops_simple = (ops * n_hot) / seconds / 1e9;

    std::cout << "n, gflops_rocblas, gflops_simple";

//  std::cout << "sec, gflops = " << seconds << ", " << gflops << std::endl; 

    if(n % 4 == 0 && incx == 1 && NB % 4 == 0 && std::is_same_v<float, T>)
    {
        rocblas_int n_mult_4 = n / 4;

        int blocks_mult_4 = (n_mult_4 - 1) / NB + 1;
        dim3 grid_mult_4(blocks_mult_4, 1, 1);
        dim3 threads_mult_4(NB, 1, 1);

//      correctness
        CHECK_HIP_ERROR(hipMemcpy(dx, hx, n * sizeof(T), hipMemcpyHostToDevice));
            hipLaunchKernelGGL(Xscal_float4, grid_mult_4, threads_mult_4, 0, 0, n_mult_4, alpha, dx);
        CHECK_HIP_ERROR(hipMemcpy(hx_calc, dx, n * sizeof(T), hipMemcpyDeviceToHost));

        if (!verify_solution(n, incx, alpha, hx_calc, hx_ref)) std::cout << "  mult4, FAIL ";

//      timing
        for (int i = 0; i < n_cold; i++) { hipLaunchKernelGGL(Xscal_float4, grid_mult_4, threads_mult_4, 0, 0, n_mult_4, alpha, dx); };
        CHECK_HIP_ERROR(hipDeviceSynchronize());
        start = std::chrono::high_resolution_clock::now();

            for (int i = 0; i < n_hot; i++)
            {
                hipLaunchKernelGGL(Xscal_float4, grid_mult_4, threads_mult_4, 0, 0, n_mult_4, alpha, dx);
            }

        CHECK_HIP_ERROR(hipDeviceSynchronize());
        end = std::chrono::high_resolution_clock::now(); dur= end - start; seconds = dur.count(); 
        gflops_mult4 = (ops * n_hot) / seconds / 1e9;

        std::cout << ", gflops_mult4";

//      std::cout << "sec, gflops = " << seconds << ", " << gflops << std::endl; 
    }

    if(n % 2 == 0 && incx == 1 && NB % 2 == 0 && std::is_same_v<double, T>)
    {
        rocblas_int n_mult_2 = n / 2;

        int blocks_mult_2 = (n_mult_2 - 1) / NB + 1;
        dim3 grid_mult_2(blocks_mult_2, 1, 1);
        dim3 threads_mult_2(NB, 1, 1);

//      correctness
        CHECK_HIP_ERROR(hipMemcpy(dx, hx, n * sizeof(T), hipMemcpyHostToDevice));
            hipLaunchKernelGGL(Xscal_double2, grid_mult_2, threads_mult_2, 0, 0, n_mult_2, alpha, dx);
        CHECK_HIP_ERROR(hipMemcpy(hx_calc, dx, n * sizeof(T), hipMemcpyDeviceToHost));

        if (!verify_solution(n, incx, alpha, hx_calc, hx_ref)) std::cout << "  mult2; FAIL ";

//      timing
        for (int i = 0; i < n_cold; i++) { hipLaunchKernelGGL(Xscal_double2, grid_mult_2, threads_mult_2, 0, 0, n_mult_2, alpha, dx); };
        CHECK_HIP_ERROR(hipDeviceSynchronize());
        start = std::chrono::high_resolution_clock::now();

            for (int i = 0; i < n_hot; i++)
            {
                hipLaunchKernelGGL(Xscal_double2, grid_mult_2, threads_mult_2, 0, 0, n_mult_2, alpha, dx);
            }

        CHECK_HIP_ERROR(hipDeviceSynchronize());
        end = std::chrono::high_resolution_clock::now(); dur= end - start; seconds = dur.count(); 
        gflops_mult2 = (ops * n_hot) / seconds / 1e9;

        std::cout << ", gflops_mult2";

//      std::cout << "sec, gflops = " << seconds << ", " << gflops << std::endl; 
    }

    std::cout << "  " << n << ", " << gflops_rocblas << ", " << gflops_simple;
    if(n % 4 == 0 && incx == 1 && NB % 4 == 0 && std::is_same_v<float, T>) std::cout << ", " << gflops_mult4;
    if(n % 2 == 0 && incx == 1 && NB % 2 == 0 && std::is_same_v<double, T>) std::cout << ", " << gflops_mult2;

    std::cout << std::endl;

    CHECK_HIP_ERROR(hipFree(dx));
    free(hx);
    free(hx_ref);
    free(hx_calc);
}

int main(int argc, char* argv[])
{
    rocblas_int n = 10240, incx = 1;
    bool verbose = false;
    char precision = 's';
    int64_t n_hot = 500;
    int64_t n_cold = 500;

    if(parse_arguments(argc, argv, n, n_hot, n_cold, verbose, precision))
    {
        show_usage(argv);
        return EXIT_FAILURE;
    }
    std::cout << "hot and cold calls = " << n_hot << ", " << n_cold;

//  if(precision == 's' || precision == 'S')
//  {
        precision = 's';
        std::cout << "----- float -----" << std::endl;
//      for (rocblas_int n = 10240; n < 102400; n += 10240)
        for (rocblas_int n = 2048000; n < 20480000; n += 2048000)
        {
            template_scal<float>(n*4, incx, n_hot, n_cold);
        }
        std::cout << "------------------------------------------------------" << std::endl;
//  }
//  else if(precision == 'd' || precision == 'D')
//  {
        precision = 'd';
        std::cout << "----- double -----" << std::endl;;
        for (rocblas_int n = 2048000; n < 20480000; n += 2048000)
        {
            template_scal<double>(n*2, incx, n_hot, n_cold);
        }
        std::cout << "------------------------------------------------------" << std::endl;
//  }
//  else if(precision == 'c' || precision == 'C')
//  {
        precision = 'c';
        std::cout << "----- rocblas_float_complex -----" << std::endl;
        for (rocblas_int n = 2048000; n < 20480000; n += 2048000)
        {
            template_scal<rocblas_float_complex>(n, incx, n_hot, n_cold);
        }
        std::cout << "------------------------------------------------------" << std::endl;
//  }
//  else if(precision == 'z' || precision == 'Z')
//  {
        precision = 'z';
        std::cout << "----- rocblas_double_complex -----" << std::endl;
        for (rocblas_int n = 2048000; n < 20480000; n += 2048000)
        {
            template_scal<rocblas_double_complex>(n, incx, n_hot, n_cold);
        }
        std::cout << "------------------------------------------------------" << std::endl;
//  }

    std::cout << std::endl;

    return 0;
}
