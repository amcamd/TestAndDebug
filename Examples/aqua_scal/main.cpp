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
                  << std::endl;
}

static int parse_arguments(
        int argc,
        char *argv[],
        rocblas_int &n, 
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

/*
template <>
void initialize_arrays<float>(rocblas_int n, rocblas_int incx, float* x, float* x_ref)
{
    std::random_device rd;  // a seed source for the random number engine
    std::mt19937 generator(rd()); // mersenne_twister_engine seeded with rd()
    std::uniform_int_distribution<int> distribution(1, 6);

    for (int i = 0; i < n; i++)
    {
        x[i*incx] = static_cast<float>(distribution(generator));
        x_ref[i*incx] = x[i*incx];
    }
}
template <>
void initialize_arrays<double>(rocblas_int n, rocblas_int incx, double* x, double* x_ref)
{
    std::random_device rd;  // a seed source for the random number engine
    std::mt19937 generator(rd()); // mersenne_twister_engine seeded with rd()
    std::uniform_int_distribution<int> distribution(1, 6);

    for (int i = 0; i < n; i++)
    {
        x[i*incx] = static_cast<double>(distribution(generator));
        x_ref[i*incx] = x[i*incx];
    }
}
*/

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
void template_scal(rocblas_int n, rocblas_int incx)
{
    T alpha = 10.0;
    T *dx, *dx_mult_4, *hx, *hx_ref;

    hx = (T*)malloc(n*sizeof(T));
    hx_ref = (T*)malloc(n*sizeof(T));

    initialize_arrays(n, incx, hx, hx_ref);
    ref_calc(n, incx, alpha, hx_ref);

    CHECK_HIP_ERROR(hipMalloc(&dx, n * sizeof(T)));
    CHECK_HIP_ERROR(hipMalloc(&dx_mult_4, n * sizeof(T)));

    CHECK_HIP_ERROR(hipMemcpy(       dx, hx, n * sizeof(T), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dx_mult_4, hx, n * sizeof(T), hipMemcpyHostToDevice));

    int blocks = (n - 1) / NB + 1;
    dim3 grid(blocks, 1, 1);
    dim3 threads(NB, 1, 1);

    std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
    std::chrono::duration<double> dur;
    double seconds = 0.0;

    start = std::chrono::high_resolution_clock::now();
    CHECK_HIP_ERROR(hipDeviceSynchronize());

        hipLaunchKernelGGL(Xscal, grid, threads, 0, 0, n, alpha, dx, 1);

    CHECK_HIP_ERROR(hipDeviceSynchronize());
    end = std::chrono::high_resolution_clock::now();
    CHECK_HIP_ERROR(hipDeviceSynchronize());
    dur= end - start;
    seconds = dur.count();

    CHECK_HIP_ERROR(hipMemcpy(hx, dx, n * sizeof(T), hipMemcpyDeviceToHost));

    verify_solution(n, incx, alpha, hx, hx_ref) ? std::cout << "PASS Xscal "
                                                : std::cout << "FAIL Xscal ";

    double ops = n;
    double gflops = ops / seconds / 1e9;

    std::cout << "sec, gflops = " << seconds << ", " << gflops << std::endl; 

    if(n % 4 == 0 && incx == 1 && NB % 4 == 0 && std::is_same_v<float, T>)
    {
        rocblas_int n_mult_4 = n / 4;

        int blocks_mult_4 = (n_mult_4 - 1) / NB + 1;
        dim3 grid_mult_4(blocks_mult_4, 1, 1);
        dim3 threads_mult_4(NB, 1, 1);

        start = std::chrono::high_resolution_clock::now();
        CHECK_HIP_ERROR(hipDeviceSynchronize());

            hipLaunchKernelGGL(Xscal_float4, grid_mult_4, threads_mult_4, 0, 0, n_mult_4, alpha, dx_mult_4);

        CHECK_HIP_ERROR(hipDeviceSynchronize());
        end = std::chrono::high_resolution_clock::now();
        CHECK_HIP_ERROR(hipDeviceSynchronize());
        dur= end - start;
        seconds = dur.count();

        CHECK_HIP_ERROR(hipMemcpy(hx, dx_mult_4, n * sizeof(T), hipMemcpyDeviceToHost));

        verify_solution(n, incx, alpha, hx, hx_ref) ? std::cout << "PASS dx_mult_4 "
                                                    : std::cout << "FAIL dx_mult_4 ";

        double ops = n;
        double gflops = ops / seconds / 1e9;

        std::cout << "sec, gflops = " << seconds << ", " << gflops << std::endl; 
    }

    CHECK_HIP_ERROR(hipFree(dx));
    CHECK_HIP_ERROR(hipFree(dx_mult_4));
    free(hx);
    free(hx_ref);
}

int main(int argc, char* argv[])
{
    rocblas_int n = 10240, incx = 1;
    bool verbose = false;
    char precision = 's';

    if(parse_arguments(argc, argv, n, verbose, precision))
    {
        show_usage(argv);
        return EXIT_FAILURE;
    }

//  if(precision == 's' || precision == 'S')
//  {
        precision = 's';
        std::cout << "precision, n*4, incx = float, " << n*4 << ", " << incx << std::endl;
        template_scal<float>(n*4, incx);
        std::cout << "------------------------------------------------------" << std::endl;
//  }
//  else if(precision == 'd' || precision == 'D')
//  {
        precision = 'd';
        std::cout << "precision, n*2, incx = double, " << n*2 << ", " << incx << std::endl;;
        template_scal<double>(n*2, incx);
        std::cout << "------------------------------------------------------" << std::endl;
//  }
//  else if(precision == 'c' || precision == 'C')
//  {
////    precision = 'c';
////    std::cout << "precision, n = rocblas_float_complex, " << n;
////    template_scal<rocblas_float_complex>(n, incx);
////    std::cout << "------------------------------------------------------" << std::endl;
//  }
//  else if(precision == 'z' || precision == 'Z')
//  {
        precision = 'z';
        std::cout << "precision, n, incx = rocblas_double_complex, " << n << ", " << incx << std::endl;
        template_scal<rocblas_double_complex>(n, incx);
        std::cout << "------------------------------------------------------" << std::endl;
//  }

    std::cout << std::endl;

    return 0;
}
