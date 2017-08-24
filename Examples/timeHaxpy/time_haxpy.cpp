#include<iostream>
#include<hip/hip_runtime.h>
#include<hip/hip_runtime_api.h>
#include "rocblas.h"

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

template <typename T>
inline rocblas_status axpy_template(rocblas_handle handle, 
        int n, T *alpha, T *xd, int incx, T *yd, int incy);

template<>
rocblas_status axpy_template<__fp16>(rocblas_handle handle,
        int n, __fp16 *alpha, __fp16 *xd, int incx, __fp16 *yd, int incy)
{
    return rocblas_haxpy(handle, n, alpha, xd, incx, yd, incy);
};

template<>
rocblas_status axpy_template<float>(rocblas_handle handle,
        int n, float *alpha, float *xd, int incx, float *yd, int incy)
{
    return rocblas_saxpy(handle, n, alpha, xd, incx, yd, incy);
};

template<>
rocblas_status axpy_template<double>(rocblas_handle handle,
        int n, double *alpha, double *xd, int incx, double *yd, int incy)
{
    return rocblas_daxpy(handle, n, alpha, xd, incx, yd, incy);
};

template <typename T> int byte_per_flop();

template<>
int byte_per_flop<__fp16>()
{
    return 2;
}

template<>
int byte_per_flop<float>()
{
    return 4;
}

template<>
int byte_per_flop<double>()
{
    return 8;
}


template <typename T>
void time_axpy(int n, int incx, int incy, T alpha, int number_inner_calls, int number_outer_tests)
{
    std::vector<T> x(n*incx), y(n*incy);

    for(int i = 0; i < x.size(); i++) x[i] = T((i+4)%4);
    for(int i = 0; i < y.size(); i++) y[i] = T(i%2); 

    int size_x = n * incx * sizeof(T);
    int size_y = n * incy * sizeof(T);
    T *xd, *yd;
    CHECK_HIP_ERROR(hipMalloc(&xd, size_x));
    CHECK_HIP_ERROR(hipMalloc(&yd, size_y));
    CHECK_HIP_ERROR(hipMemcpy(xd, x.data(), size_x, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(yd, y.data(), size_y, hipMemcpyHostToDevice));

    rocblas_handle handle;
    CHECK_ROCBLAS_ERROR(rocblas_create_handle(&handle));

    // detect possible error before timing
    CHECK_ROCBLAS_ERROR(axpy_template<T>(handle, n, &alpha, xd, incx, yd, incy));

    std::vector<double> times(number_outer_tests);

    double min_seconds = std::numeric_limits<double>::max();
    double max_seconds = std::numeric_limits<double>::min();
    double sum_seconds = 0.0;
    double seconds = 0.0;

    std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
    std::chrono::duration<double> dur;

    for(int i = 0; i < number_outer_tests; i++)
    {
        hipDeviceSynchronize();
        start = std::chrono::high_resolution_clock::now();

        for(int i = 0; i < number_inner_calls; i++) 
        {
            axpy_template<T>(handle, n, &alpha, xd, incx, yd, incy);
        }

        hipDeviceSynchronize();
        end = std::chrono::high_resolution_clock::now();
        dur= end - start;
        seconds = dur.count();

        min_seconds = min_seconds < seconds ? min_seconds : seconds;
        max_seconds = max_seconds > seconds ? max_seconds : seconds;
        sum_seconds = sum_seconds + seconds;

        times[i] = seconds;
    }
    double ave_seconds = sum_seconds / (double) number_outer_tests;

    double flop = ((double) n) * 2.0 * ((double)number_inner_calls);
    double ld_byte = ((double) n) * 2.0 * byte_per_flop<T>() * ((double)number_inner_calls);
    double st_byte = ((double) n) * 1.0 * byte_per_flop<T>() * ((double)number_inner_calls);
//  double ld_byte = ((double) n) * 2.0 * 2.0 * ((double)number_inner_calls);
//  double st_byte = ((double) n) * 1.0 * 2.0 * ((double)number_inner_calls);

    double max_gflops = flop / min_seconds / 1e9;
    double min_gflops = flop / max_seconds / 1e9;
    double ave_gflops = flop / ave_seconds / 1e9;

    double max_ld_gbyte_s = ld_byte / min_seconds / 1e9;
    double min_ld_gbyte_s = ld_byte / max_seconds / 1e9;
    double ave_ld_gbyte_s = ld_byte / ave_seconds / 1e9;

    double max_st_gbyte_s = st_byte / min_seconds / 1e9;
    double min_st_gbyte_s = st_byte / max_seconds / 1e9;
    double ave_st_gbyte_s = st_byte / ave_seconds / 1e9;

    //calculate relative standard deviation (rsd). Also called coefficient of variation
    double rsd_seconds = 0.0, rsd_gflops = 0.0;
    for(int i = 0; i < number_outer_tests; i++) 
    {
        rsd_seconds += (times[i] - ave_seconds) * (times[i] - ave_seconds) ;
    }
    rsd_seconds = rsd_seconds / (double) number_outer_tests;
    rsd_seconds = sqrt(rsd_seconds) / ave_seconds * 100.0;

    printf("max,ave,min,rsd_seconds = %f, %f, %f, %.1f%%\n", 
            max_seconds, ave_seconds, min_seconds, rsd_seconds);
    printf("min,ave,max_gflop/sec = %.0f, %.0f, %.0f\n", min_gflops, ave_gflops, max_gflops);
    printf("min,ave,max_ld_gB/sec = %.0f, %.0f, %.0f\n", min_ld_gbyte_s, ave_ld_gbyte_s, max_ld_gbyte_s);
    printf("min,ave,max_st_gB/sec = %.0f, %.0f, %.0f\n", min_st_gbyte_s, ave_st_gbyte_s, max_st_gbyte_s);
    
    CHECK_HIP_ERROR(hipFree(xd));
    CHECK_HIP_ERROR(hipFree(yd));
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
    std::cout << "n, incx, incy = " << n << ", " << incx << ", " << incy << std::endl;
        
    int number_inner_calls = 100;
    int number_outer_tests = 10;

    printf("number_inner_calls, number_outer_tests= %d, %d\n", number_inner_calls, number_outer_tests);

    printf("--- __fp16 -----------------------------------------------\n");
    time_axpy<__fp16>(n, incx, incy, alpha, number_inner_calls, number_outer_tests);
    printf("--- float  -----------------------------------------------\n");
    time_axpy<float>(n, incx, incy, alpha, number_inner_calls, number_outer_tests);
    printf("--- double -----------------------------------------------\n");
    time_axpy<double>(n, incx, incy, alpha, number_inner_calls, number_outer_tests);
}

