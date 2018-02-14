#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <vector>
#include <math.h>
#include <chrono>
#include <limits>
#include <unistd.h>
#include "rocblas.h"
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

#define M_DIM 512
#define N_DIM 512
#define K_DIM 512
#define TRANS_A rocblas_operation_none
#define TRANS_B rocblas_operation_none
#define ALPHA 1.1
#define BETA 0.3
#define P_MAX 8
#define PERFORMANCE_TEST true
#define CORRECTNESS_TEST false
#define SLEEP_MULTIPLIER 0

#define CHECK_HIP_ERROR(error) \
    if (error != hipSuccess) { \
      fprintf(stderr, "Hip error: '%s'(%d) at %s:%d\n", hipGetErrorString(error), error,__FILE__, __LINE__); \
      exit(EXIT_FAILURE);\
    }

#define CHECK_ROCBLAS_ERROR(error) \
     if (error != rocblas_status_success) { \
        fprintf(stderr, "rocBLAS error: "); \
        if(error == rocblas_status_invalid_handle)fprintf(stderr, "rocblas_status_invalid_handle"); \
        if(error == rocblas_status_not_implemented )fprintf(stderr, " rocblas_status_not_implemented"); \
        if(error == rocblas_status_invalid_pointer)fprintf(stderr, "rocblas_status_invalid_pointer"); \
        if(error == rocblas_status_invalid_size)fprintf(stderr, "rocblas_status_invalid_size"); \
        if(error == rocblas_status_memory_error)fprintf(stderr, "rocblas_status_memory_error"); \
        if(error == rocblas_status_internal_error)fprintf(stderr, "rocblas_status_internal_error"); \
        fprintf(stderr, "\n"); \
        exit(EXIT_FAILURE); \
     }

typedef enum output_mode_
{
    terse = 0,
    verbose = 1
} output_mode;

using namespace std;

void usage(char *argv[])
{
    printf("Usage: %s\n", argv[0]);
    printf(" -m<gemm m, default %d>\n", M_DIM);
    printf(" -n<gemm n, default %d>\n", N_DIM);
    printf(" -k<gemm k, default %d>\n", K_DIM);
    printf(" -t<gemm NN, NT, TN, TT, default NN>\n");
    printf(" -a<gemm lda, default %d>\n", M_DIM);
    printf(" -b<gemm ldb, default %d>\n", K_DIM);
    printf(" -c<gemm ldc, default %d>\n", M_DIM);
    printf(" -o<output verbose or terse: v or t, default v\n");
    printf(" -f<output first line: y or n, default y\n");
    printf(" -s<sleep_multiplier, default %d>\n", SLEEP_MULTIPLIER);
    exit (8);
}

int parse_args(int argc, char *argv[], int &M, int &N, int &K, int &lda, int &ldb, int &ldc,
                rocblas_operation &transA, rocblas_operation &transB, output_mode &output, bool &first,
                useconds_t sleep_multiplier)
{
    while (argc > 1)
    {
        if (argv[1][0] == '-')
        {
            switch (argv[1][1])
            {
                case 't':
                    if(strcmp(&argv[1][2], "NN") == 0)
                    {
                        transA = rocblas_operation_none;
                        transB = rocblas_operation_none;
                    }
                    else if(strncmp(&argv[1][2], "NT", 2) == 0)
                    {
                        transA = rocblas_operation_none;
                        transB = rocblas_operation_transpose;
                    }
                    else if(strncmp(&argv[1][2], "TN", 2) == 0)
                    {
                        transA = rocblas_operation_transpose;
                        transB = rocblas_operation_none;
                    }
                    else if(strncmp(&argv[1][2], "TT", 2) == 0)
                    {
                        transA = rocblas_operation_transpose;
                        transB = rocblas_operation_transpose;
                    }
                    break;
                case 'o':
                    if(strcmp(&argv[1][2], "v") == 0) 
                    {
                        output = verbose;
                    } 
                    else if(strcmp(&argv[1][2], "t") == 0) 
                    {
                        output = terse;
                    }
                    break;
                case 'f':
                    if(strcmp(&argv[1][2], "y") == 0) 
                    {
                        first = true;
                    } 
                    else if(strcmp(&argv[1][2], "n") == 0) 
                    {
                        first = false;
                    }
                    break;
                case 'm':
                    M = atoi(&argv[1][2]);
                    break;
                case 'n':
                    N = atoi(&argv[1][2]);
                    break;
                case 'k':
                    K = atoi(&argv[1][2]);
                    break;
                case 'a':
                    lda = atoi(&argv[1][2]);
                    break;
                case 'b':
                    ldb = atoi(&argv[1][2]);
                    break;
                case 'c':
                    ldc = atoi(&argv[1][2]);
                    break;
                case 's':
                    sleep_multiplier = (useconds_t)(atoi(&argv[1][2]));
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

template <typename T>
void printMatrix(const char* name, T* A, rocblas_int m, rocblas_int n, rocblas_int lda)
{
    printf("---------- %s ----------\n", name);
    for( int i = 0; i < m; i++)
    {
        for( int j = 0; j < n; j++)
        {
            printf("%f ",A[i + j * lda]);
        }
        printf("\n");
    }
}

template <typename T>
void mat_mat_mult(T alpha, T beta, int M, int N, int K, vector<T> A, int As1, int As2, 
                  vector<T> B, int Bs1, int Bs2, vector<T> & C, int Cs1, int Cs2)
{
    for(int i1=0; i1<M; i1++)
    {
        for(int i2=0; i2<N; i2++)
        {
            T t = 0.0;
            for(int i3=0; i3<K; i3++)
            {
                t +=  A[i1 * As1 + i3 * As2] * B[i3 * Bs1 + i2 * Bs2]; 
            }
            C[i1*Cs1 +i2*Cs2] = beta * C[i1*Cs1+i2*Cs2] + alpha * t ;
        }
    }
}

int element_check(int M, int N, int ldc, float tolerance, vector<float>hC, vector<float>hC_copy)
{
    float error = 0;
    for(rocblas_int i1=0; i1<M; i1++)
    {
        for(rocblas_int i2=0; i2<N; i2++)
        {
            error = fabs(hC[i1+i2*ldc] - hC_copy[i1+i2*ldc]);
            if(error != error || error > tolerance)
            {
              printf("error %d,%d: %E  CPU=%E, GPU=%E\n",i1,i2,error,hC[i1+i2*ldc],hC_copy[i1+i2*ldc]);
              break;
            }
        }
        if(error != error || error > tolerance) break;
    }

    if(error != error || error > tolerance)
    {
        return 1;
    }
    else
    {
        return 0;
    }
}

double error_norm(int n1, int n2, vector<float> c_float, vector<double> c_double)
{
    double frobenius_norm_error = 0.0;
    for (int i = 0; i < n1*n2; i++)
    {
        frobenius_norm_error += (c_double[i] - (double)c_float[i]) * (c_double[i] - (double)c_float[i]);
    }
    return(sqrt(frobenius_norm_error));
}

template <typename T>
T frobenius_norm(int n1, int n2, vector<T> a)
{
    T norm;
    for (int i = 0; i < n1*n2; i++)
    {
        norm += a[i] * a[i];
    }
    return sqrt(norm);
}

int norm_check(int M, int N, double tolerance, vector<float> hc, vector<double>hc64)
{
    float eps = std::numeric_limits<float>::epsilon();
    double frobenius_norm_error = error_norm(M, N, hc, hc64);
    double frobenius_norm_c64 = frobenius_norm<double>(M, N, hc64);
//  printf("frobenius_norm_error, frobenius_norm_c = %E, %E\n", frobenius_norm_error, frobenius_norm_c64);
    printf("(frobenius_norm_error/frobenius_norm_c) / eps = %E\n", frobenius_norm_error/frobenius_norm_c64/eps);
    double error = (frobenius_norm_error/frobenius_norm_c64) / eps;
    if (error != error || error > tolerance)
    {
        return 1;
    }
    else
    {
        return 0;
    }
}

int main(int argc, char *argv[])
{
    rocblas_int M = M_DIM;  float alpha = ALPHA;
    rocblas_int N = N_DIM;  float beta  = BETA;
    rocblas_int K = K_DIM; 
    rocblas_int lda = 0, sizeOfA, As1, As2;  rocblas_operation transA = TRANS_A; 
    rocblas_int ldb = 0, sizeOfB, Bs1, Bs2;  rocblas_operation transB = TRANS_B; 
    rocblas_int ldc = 0, sizeOfC, Cs1, Cs2;
    output_mode output = verbose;
    useconds_t sleep_multiplier = SLEEP_MULTIPLIER; 
    bool first = true;



    int deviceId;
    hipCtxGetDevice(&deviceId);
    hipDeviceProp_t deviceProperties;
    hipGetDeviceProperties(&deviceProperties, deviceId);
    std::string name = deviceProperties.name;
    std::cout << "name =<" << name << ">" << std::endl;
    if (name == "Device 6860")
    {
        std::cout << "--- name = 6860 ---" << std::endl;
    }
    else if (name == "Fiji [Radeon R9 FURY / NANO Series]")
    {
        std::cout << "--- name = Fiji [Radeon R9 FURY / NANO Series] ---" << std::endl;
    }
    else if (name == "Device 6863")
    {
        std::cout << "--- name = 6863 ---" << std::endl;
    }
    else
    {
        std::cout << "--- name = default ---" << std::endl;
    }










    if( parse_args(argc, argv, M, N, K, lda, ldb, ldc, transA, transB, output, first, sleep_multiplier))
    {
        usage(argv);
        return -1;
    }

    if (first == true)
    {
        if (output == verbose)
        {
            if (M==N && M==K && M==lda && M==ldb && M==ldc)
            {
                printf("M==N && M==K && M==lda && M==ldb && M==ldc\n");
                printf("transA_transB,m,number_inner_iterations,number_outer_iterations,min_gflops,ave_gflops,max_gflops,rsd_gflops%%\n");
            }
            else
            {
                printf("transA_transB,m,n,k,lda,ldb,ldc,number_inner_iterations,number_outer_iterations,min_gflops,ave_gflops,max_gflops,rsd_gflops%%\n");
            }
        }
        if (output == terse)
        {
            if (M==N && M==K && M==lda && M==ldb && M==ldc)
            {
                printf("M==N && M==K && M==lda && M==ldb && M==ldc\n");
                printf("transA_transB,m,max_gflops\n");
            }
            else
            {
                printf("transA_transB,m,n,k,lda,ldb,ldc,max_gflops\n");
            }
        }
    }

    ldc = M;             // leading dimension of C (gemm argument)
    sizeOfC = N * ldc;   // number of elements for memory allocation
    Cs1 = 1;             // stride in first index
    Cs2 = ldc;           // stride in second index
    // set leading dimension and strides depending on transA and transB
    if( transA == rocblas_operation_none)
    {
        lda = lda >= M ? lda : M; As1 = 1; As2 = lda; sizeOfA = K * lda; printf("N");
    }
    else
    {
        lda = lda >= K ? lda : K; As1 = lda; As2 = 1; sizeOfA = M * lda; printf("T");
    }
    if( transB == rocblas_operation_none)
    {
        ldb = ldb >= K ? ldb : K; Bs1 = 1; Bs2 = ldb; sizeOfB = N * ldb; printf("N,");
    }
    else
    {
        ldb = ldb >= N ? ldb : N; Bs1 = ldb; Bs2 = 1; sizeOfB = K * ldb; printf("T,");
    }

    if (output == verbose)
    {
        if (M==N && M==K && M==lda && M==ldb && M==ldc)
        {
            printf("%d,",M);
        }
        else
        {
            printf("%d,%d,%d,%d,%d,%d,",M, N, K, lda, ldb, ldc);
        }
    }
    else
    {
        printf("%d,",M);
    }

    vector<float> hA(sizeOfA), hB(sizeOfB), hC(sizeOfC), hC_copy(sizeOfC);
    vector<double> hA64(sizeOfA), hB64(sizeOfB), hC64(sizeOfC);
    float *dA, *dB, *dC;
    float element_tolerance =10;
    double norm_tolerance = 80;

    rocblas_handle handle;
    CHECK_ROCBLAS_ERROR(rocblas_create_handle(&handle));

    // allocate memory on device
    CHECK_HIP_ERROR(hipMalloc(&dA, sizeOfA * sizeof(float)));
    CHECK_HIP_ERROR(hipMalloc(&dB, sizeOfB * sizeof(float)));
    CHECK_HIP_ERROR(hipMalloc(&dC, sizeOfC * sizeof(float)));

    // Initial Data on CPU,
    srand(1);
    for( int i = 0; i < sizeOfA; ++i ) { hA[i] = rand() % 17; hA64[i] = (double)hA[i]; }
    for( int i = 0; i < sizeOfB; ++i ) { hB[i] = rand() % 17; hB64[i] = (double)hB[i]; }
    for( int i = 0; i < sizeOfC; ++i ) { hC[i] = rand() % 17; hC64[i] = (double)hC[i]; }

    // save a copy in hA
    hC_copy = hC;

    CHECK_HIP_ERROR(hipMemcpy(dA, hA.data(), sizeof(float) * sizeOfA, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dB, hB.data(), sizeof(float) * sizeOfB, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dC, hC.data(), sizeof(float) * sizeOfC, hipMemcpyHostToDevice));

    if(CORRECTNESS_TEST)
    {
        CHECK_ROCBLAS_ERROR(rocblas_sgemm(handle, transA, transB, M, N, K, &alpha, dA, lda, dB, 
                               ldb, &beta, dC, ldc));

        // copy output from device memory to host memory
        CHECK_HIP_ERROR(hipMemcpy(hC.data(), dC, sizeof(float) * sizeOfC, hipMemcpyDeviceToHost));

        mat_mat_mult<float>(alpha, beta, M, N, K, hA, As1, As2, hB, Bs1, Bs2, hC_copy, 1, ldc);
        mat_mat_mult<double>((double)alpha, (double)beta, M, N, K, hA64, As1, As2, hB64, Bs1, Bs2, hC64, 1, ldc);

//      printMatrix<float>("calculated matrix hC", hC.data(), min(P_MAX,M), min(P_MAX,N), ldc);
//      printMatrix<float>("reference  matrix hC_copy", hC_copy.data(), min(P_MAX,M), min(P_MAX,N), ldc);

        if( element_check(M, N, ldc, element_tolerance, hC, hC_copy))
        {
            printf("GEMM Failed element wise test !\n"); 
        }
        else
        {
            printf("GEMM Success in element wise test!\n");
        }

        if( norm_check(M, N, norm_tolerance, hC, hC64))
        {
            printf("GEMM Failed norm test !\n");
        } 
        else
        {
            printf("GEMM passed norm test!\n");
        }
    }

    if(PERFORMANCE_TEST)
    {
        std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
        std::chrono::duration<double> dur;
        hipEvent_t hipStart, hipStop;

        hipEventCreate(&hipStart);
        hipEventCreate(&hipStop);
        float milliseconds = 0.0;
        double seconds = 0.0;
        useconds_t sleep_micro_sec; 

//      time one rocblas_sgemm call for warmup
        int ninner = 1;

#define CHRON_TIMER true
#if CHRON_TIMER == true
//      printf("CHRON_TIMER == true\n");
        hipDeviceSynchronize();
        start = std::chrono::high_resolution_clock::now();
        hipDeviceSynchronize();
#else
//      printf("CHRON_TIMER == false\n");
        hipEventRecord(hipStart);
#endif

        for(int i = 0; i < ninner; i++)
        {
            CHECK_ROCBLAS_ERROR(rocblas_sgemm(handle, transA, transB, M, N, K, &alpha, dA, lda, dB, 
                               ldb, &beta, dC, ldc));
        }

#if CHRON_TIMER == true
        hipDeviceSynchronize();
        end = std::chrono::high_resolution_clock::now();
        hipDeviceSynchronize();
        dur= end - start;
        seconds = dur.count();
#else
        hipEventRecord(hipStop);
        hipEventElapsedTime(&milliseconds, hipStart, hipStop);
        seconds = (double) milliseconds / 1000.0;
#endif
        sleep_micro_sec = (unsigned int)(seconds * 1000000.0);
        if(sleep_multiplier > 0)
        {
            usleep(sleep_micro_sec * sleep_multiplier);
        }

//      if one call takes less than 1 second, time another 4 calls for warmup
        if(seconds < 1)
        {
            ninner = 4;

#if CHRON_TIMER == true
            hipDeviceSynchronize();
            start = std::chrono::high_resolution_clock::now();
            hipDeviceSynchronize();
#else
            hipEventRecord(hipStart);
#endif

            for(int i = 0; i < ninner; i++)
            {
                CHECK_ROCBLAS_ERROR(rocblas_sgemm(handle, transA, transB, M, N, K, &alpha, dA, lda, dB, 
                                   ldb, &beta, dC, ldc));
            }

#if CHRON_TIMER == true
            hipDeviceSynchronize();
            end = std::chrono::high_resolution_clock::now();
            hipDeviceSynchronize();
            dur= end - start;
            seconds = dur.count();
#else
            hipEventRecord(hipStop);
            hipEventElapsedTime(&milliseconds, hipStart, hipStop);
            seconds = (double) milliseconds / 1000.0;
#endif
            sleep_micro_sec = (unsigned int)(seconds * 1000000.0);
            if(sleep_multiplier > 0)
            {
                usleep(sleep_micro_sec * sleep_multiplier);
            }

        }

        // number of inner iterations to run for 0.05 sec, limit to between 1 and 10
        ninner = (0.05 / seconds) * ninner;
        ninner = 10 < ninner ? 10 : ninner;
        ninner = 1 > ninner ? 1 : ninner;

        // number of outer iterations required to run for 50 second, limit to between 100 and 1000
        int number_iterations = 50.0 / seconds; 
        number_iterations = 100 > number_iterations ? 100 : number_iterations;
        number_iterations = 1000 < number_iterations ? 1000 : number_iterations;

        vector<double> times(number_iterations);

        double min_seconds = numeric_limits<double>::max();
        double max_seconds = numeric_limits<double>::min();
        double sum_seconds = 0.0;
        for(int i = 0; i < number_iterations; i++)
        {
#if CHRON_TIMER == true
            hipDeviceSynchronize();
            start = std::chrono::high_resolution_clock::now();
            hipDeviceSynchronize();
#else
            hipEventRecord(hipStart);
#endif

            for(int i = 0; i < ninner; i++) {
                rocblas_sgemm(handle, transA, transB, M, N, K, &alpha, dA, lda, dB, ldb, &beta, dC, ldc);
            }

#if CHRON_TIMER == true
            hipDeviceSynchronize();
            end = std::chrono::high_resolution_clock::now();
            hipDeviceSynchronize();
            dur= (end - start);
            seconds = dur.count() / ninner;
#else
            hipEventRecord(hipStop);
            hipEventElapsedTime(&milliseconds, hipStart, hipStop);
            seconds = (double) milliseconds / 1000.0 / ninner;
#endif
            sleep_micro_sec = (unsigned int)(seconds * ninner * 1000000.0);
            usleep(sleep_micro_sec * 5);
            if(sleep_multiplier > 0)
            {
                usleep(sleep_micro_sec * sleep_multiplier);
            }

            min_seconds = min_seconds < seconds ? min_seconds : seconds;
            max_seconds = max_seconds > seconds ? max_seconds : seconds;
            sum_seconds = sum_seconds + seconds;

            times[i] = seconds;
        }
        double ave_seconds = sum_seconds / (double) number_iterations;
        double ops = (double)(M) * (double)(N) * (double)(K) * 2.0;
        double max_gflops = ops / min_seconds / 1e9;
        double min_gflops = ops / max_seconds / 1e9;
        double ave_gflops = ops / ave_seconds / 1e9;
        //calculate relative standard deviation (rsd). Also called coefficient of variation
        double rsd_seconds = 0.0, rsd_gflops = 0.0;
        for(int i = 0; i < number_iterations; i++) {
            rsd_seconds += (times[i] - ave_seconds) * (times[i] - ave_seconds) ;
            rsd_gflops += (ops / times[i] / 1.e9 - ave_gflops) * (ops / times[i] / 1.e9 - ave_gflops) ;
        }
        rsd_seconds = rsd_seconds / (double) number_iterations;
        rsd_gflops = rsd_gflops / (double) number_iterations;
        rsd_seconds = sqrt(rsd_seconds) / ave_seconds * 100.0;
        rsd_gflops = sqrt(rsd_gflops) / ave_gflops * 100.0;
//      printf("number_iterations, max,ave,min,rsd_seconds= %d, %f, %f, %f, %.1f%%\n", number_iterations, max_seconds, ave_seconds, min_seconds, rsd_seconds );

        if (output == verbose)
        {
            printf("%d,%d,%.0f,%.0f,%.0f,%.1f,", ninner, number_iterations, min_gflops, ave_gflops, max_gflops, rsd_gflops);
        }
        if (output == terse)
        {
            printf("%.0f,", max_gflops);
        }
    }

    CHECK_HIP_ERROR(hipFree(dA));
    CHECK_HIP_ERROR(hipFree(dB));
    CHECK_HIP_ERROR(hipFree(dC));
    CHECK_ROCBLAS_ERROR(rocblas_destroy_handle(handle));

    return 0;
}
