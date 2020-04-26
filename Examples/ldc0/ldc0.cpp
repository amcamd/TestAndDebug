/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include <iomanip>
//   #include <random>
#include <vector>
#include <cmath>
#include <math.h>
#include <immintrin.h>
#include <iostream>
#include <hip/hip_runtime.h>
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

template <typename T>
void print_matrix(
    const char* name, std::vector<T>& A, rocblas_int m, rocblas_int n, rocblas_int lda)
{
    printf("---------- %s ----------\n", name);
    int max_size = 3;
    for(int i = 0; i < m && i < max_size; i++)
    {
        for(int j = 0; j < n && j < max_size; j++)
        {
            std::cout << std::setw(4) << A[i + j * lda] << " ";
        }
        std::cout << "\n";
    }
}

template <typename T>
void mat_mat_mult(T alpha,
                  T beta,
                  int M,
                  int N,
                  int K,
                  std::vector<T>& A,
                  int As1,
                  int As2,
                  std::vector<T>& B,
                  int Bs1,
                  int Bs2,
                  std::vector<T>& C,
                  int Cs1,
                  int Cs2,
                  std::vector<T>& D,
                  int Ds1,
                  int Ds2)
{
    for(int i1 = 0; i1 < M; i1++)
    {
        for(int i2 = 0; i2 < N; i2++)
        {
            float t = 0.0;
            for(int i3 = 0; i3 < K; i3++)
            {
                t += A[i1 * As1 + i3 * As2] * B[i3 * Bs1 + i2 * Bs2];
            }
            D[i1 * Ds1 + i2 * Ds2] = beta * C[i1 * Cs1 + i2 * Cs2] + alpha * t;
        }
    }
}

static void show_usage(char* argv[])
{
    std::cerr << "Usage: " << argv[0] << " <options>\n"
              << "options:\n"
              << "\t-h, --help\t\t\t\tShow this help message\n"
              << "\t-v, --verbose\t\t\t\tverbose output\n"
              << "\t-m \t\t\tm\t\trocblas_gemm_ex argument m\n"
              << "\t-n \t\t\tn\t\trocblas_gemm_ex argument n\n"
              << "\t-k \t\t\tk \t\trocblas_gemm_ex argument k\n"
              << "\t--lda \t\t\tlda \t\trocblas_gemm_ex argument lda\n"
              << "\t--ldb \t\t\tldb \t\trocblas_gemm_ex argument ldb\n"
              << "\t--ldc \t\t\tldc \t\trocblas_gemm_ex argument ldc\n"
              << "\t--trans_a \t\ttrans_a \tn, N, t, or T\n"
              << "\t--trans_b \t\ttrans_b \tn, N, t, or T\n"
              << "\t--alpha \t\talpha \t\trocblas_gemm_ex argument alpha\n"
              << "\t--beta \t\t\tbeta \t\trocblas_gemm_ex argument beta\n"
              << "\t--header \t\theader \t\tprint header for output\n"
              << std::endl;
}

static int parse_arguments(int argc,
                           char* argv[],
                           int& m,
                           int& n,
                           int& k,
                           int& lda,
                           int& ldb,
                           int& ldc,
                           int& ldd,
                           float& alpha,
                           float& beta,
                           rocblas_operation& trans_a,
                           rocblas_operation& trans_b,
                           bool& header,
                           bool& verbose)
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
                else if(arg == "--header")
                {
                    header = true;
                }
                else if((arg == "-m") && (i + 1 < argc))
                {
                    m = atoi(argv[++i]);
                }
                else if((arg == "-n") && (i + 1 < argc))
                {
                    n = atoi(argv[++i]);
                }
                else if((arg == "-k") && (i + 1 < argc))
                {
                    k = atoi(argv[++i]);
                }
                else if((arg == "--lda") && (i + 1 < argc))
                {
                    lda = atoi(argv[++i]);
                }
                else if((arg == "--ldb") && (i + 1 < argc))
                {
                    ldb = atoi(argv[++i]);
                }
                else if((arg == "--ldc") && (i + 1 < argc))
                {
                    ldc = atoi(argv[++i]);
                }
                else if((arg == "--ldd") && (i + 1 < argc))
                {
                    ldd = atoi(argv[++i]);
                }
                else if((arg == "--alpha") && (i + 1 < argc))
                {
                    alpha = atof(argv[++i]);
                }
                else if((arg == "--beta") && (i + 1 < argc))
                {
                    beta = atof(argv[++i]);
                }
                else if((arg == "--trans_a") && (i + 1 < argc))
                {
                    ++i;
                    if(strncmp(argv[i], "N", 1) == 0 || strncmp(argv[i], "n", 1) == 0)
                    {
                        trans_a = rocblas_operation_none;
                    }
                    else if(strncmp(argv[i], "T", 1) == 0 || strncmp(argv[i], "t", 1) == 0)
                    {
                        trans_a = rocblas_operation_transpose;
                    }
                    else
                    {
                        std::cerr << "error with " << arg << std::endl;
                        std::cerr << "do not recognize value " << argv[i];
                        return EXIT_FAILURE;
                    }
                }
                else if((arg == "--trans_b") && (i + 1 < argc))
                {
                    ++i;
                    if(strncmp(argv[i], "N", 1) == 0 || strncmp(argv[i], "n", 1) == 0)
                    {
                        trans_b = rocblas_operation_none;
                    }
                    else if(strncmp(argv[i], "T", 1) == 0 || strncmp(argv[i], "t", 1) == 0)
                    {
                        trans_b = rocblas_operation_transpose;
                    }
                    else
                    {
                        std::cerr << "error with " << arg << std::endl;
                        std::cerr << "do not recognize value " << argv[i];
                        return EXIT_FAILURE;
                    }
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

bool bad_argument(rocblas_operation trans_a,
                  rocblas_operation trans_b,
                  rocblas_int m,
                  rocblas_int n,
                  rocblas_int k,
                  rocblas_int lda,
                  rocblas_int ldb,
                  rocblas_int ldc,
                  rocblas_int ldd)
{
    bool argument_error = false;
    if((trans_a == rocblas_operation_none) && (lda < m))
    {
        argument_error = true;
        std::cerr << "ERROR: bad argument lda = " << lda << " < " << m << std::endl;
    }
    if((trans_a == rocblas_operation_transpose) && (lda < k))
    {
        argument_error = true;
        std::cerr << "ERROR: bad argument lda = " << lda << " < " << k << std::endl;
    }
    if((trans_b == rocblas_operation_none) && (ldb < k))
    {
        argument_error = true;
        std::cerr << "ERROR: bad argument ldb = " << ldb << " < " << k << std::endl;
    }
    if((trans_b == rocblas_operation_transpose) && (ldb < n))
    {
        argument_error = true;
        std::cerr << "ERROR: bad argument ldb = " << ldb << " < " << n << std::endl;
    }
    if(ldc < m)
    {
	//allow ldc == 0
//      argument_error = true;
        std::cerr << "WARNING: argument ldc = " << ldc << " < " << m << std::endl;
    }
    if(ldd < m)
    {
        argument_error = true;
        std::cerr << "ERROR: bad argument ldd = " << ldd << " < " << m << std::endl;
    }
    return argument_error;
}

template <typename T>
void initialize_a_b_c_d(std::vector<T>& ha,
                      rocblas_int size_a,
                      std::vector<T>& hb,
                      rocblas_int size_b,
                      std::vector<T>& hc,
                      rocblas_int size_c,
		      rocblas_int ldc,
		      rocblas_int m,
                      std::vector<T>& hd,
                      rocblas_int size_d)

{
    for(int i = 0; i < size_a; ++i)
    {
        ha[i] = T(rand() % 3) + 1;
    }
    for(int i = 0; i < size_b; ++i)
    {
        hb[i] = T(rand() % 3) + 1;
    }
    if (ldc != 0)
    {
        for(int i = 0; i < size_c; ++i)
        {
            hc[i] = T(rand() % 3) + 1;
        }
    }
    else
    {
	// fill part of c that should not be accessed with signaling_NaN
        for(int i = 0; i < size_c; ++i)
        {
            hc[i] = std::numeric_limits<float>::signaling_NaN();
        }
        for(int i = 0; i < m; ++i)
        {
            hc[i] = T(rand() % 3) + 1;
        }
    }
    for(int i = 0; i < size_d; ++i)
    {
        hd[i] = T(rand() % 3) + 1;
    }
}

int main(int argc, char* argv[])
{
    rocblas_operation trans_a = rocblas_operation_none;
    rocblas_operation trans_b = rocblas_operation_none;

    rocblas_int m = 128, lda = 128;
    rocblas_int n = 128, ldb = 128;
    rocblas_int k = 128, ldc = 128, ldd = 128;

    float alpha = 2;
    float beta  = 3;

    constexpr rocblas_datatype a_type       = rocblas_datatype_f32_r;
    constexpr rocblas_datatype b_type       = rocblas_datatype_f32_r;
    constexpr rocblas_datatype c_type       = rocblas_datatype_f32_r;
    constexpr rocblas_datatype d_type       = rocblas_datatype_f32_r;
    constexpr rocblas_datatype compute_type = rocblas_datatype_f32_r;

    rocblas_gemm_algo algo = rocblas_gemm_algo_standard;
    int32_t solution_index = 0;
    uint32_t flags         = 0;
    size_t* workspace_size = 0;
    void* workspace        = 0;

    bool verbose = false;
    bool header  = false;

    if(parse_arguments(
           argc, argv, m, n, k, lda, ldb, ldc, ldd, alpha, beta, trans_a, trans_b, header, verbose))
    {
        show_usage(argv);
        return EXIT_FAILURE;
    }

    if(bad_argument(trans_a, trans_b, m, n, k, lda, ldb, ldc, ldd))
    {
        show_usage(argv);
        return EXIT_FAILURE;
    }

    if(header)
    {
        std::cout << "transAB,M,N,K,lda,ldb,ldc,ldd,alpha,beta,"
                     "result,error";
        std::cout << std::endl;
    }

    int size_a, a_stride_1, a_stride_2;
    int size_b, b_stride_1, b_stride_2;
    // make c 2x needed size when ldc == 0 and fill unused part with signaling_NaN
    int size_c = ldc == 0 ? n * 2 : ldc * n; std::cout << "size_c = " << size_c << std::endl;
    int size_d = ldd * n;
    if(trans_a == rocblas_operation_none)
    {
        std::cout << "N";
        a_stride_1 = 1;
        a_stride_2 = lda;
        size_a     = lda * k;
    }
    else
    {
        std::cout << "T";
        a_stride_1 = lda;
        a_stride_2 = 1;
        size_a     = lda * m;
    }
    if(trans_b == rocblas_operation_none)
    {
        std::cout << "N, ";
        b_stride_1 = 1;
        b_stride_2 = ldb;
        size_b     = ldb * n;
    }
    else
    {
        std::cout << "T, ";
        b_stride_1 = ldb;
        b_stride_2 = 1;
        size_b     = ldb * k;
    }

    std::cout << m << ", " << n << ", " << k << ", " << lda << ", " << ldb << ", " << ldc << ", "
              << ldd << ", " << alpha << ", " << beta << ", ";

    // Naming: da is in GPU (device) memory. ha is in CPU (host) memory
    std::vector<float> ha(size_a);
    std::vector<float> hb(size_b);
    std::vector<float> hc(size_c);
    std::vector<float> hd(size_d);
    std::vector<float> hd_gold(size_d);

    // initial data on host
    initialize_a_b_c_d(ha, size_a, hb, size_b, hc, size_c, ldc, m, hd, size_d);

    if(verbose)
    {
        printf("\n");
        if(trans_a == rocblas_operation_none)
        {
            print_matrix<float>("ha initial", ha, m, k, lda);
        }
        else
        {
            print_matrix<float>("ha initial", ha, k, m, lda);
        }
        if(trans_b == rocblas_operation_none)
        {
            print_matrix<float>("hb initial", hb, k, n, ldb);
        }
        else
        {
            print_matrix<float>("hb initial", hb, n, k, ldb);
        }
        print_matrix<float>("hc initial", hc, m, n, ldc);
        print_matrix<float>("hd initial", hd, m, n, ldd);
    }

    // allocate memory on device
    float *da, *db, *dc, *dd;
    CHECK_HIP_ERROR(hipMalloc(&da, size_a * sizeof(float)));
    CHECK_HIP_ERROR(hipMalloc(&db, size_b * sizeof(float)));
    CHECK_HIP_ERROR(hipMalloc(&dc, size_c * sizeof(float)));
    CHECK_HIP_ERROR(hipMalloc(&dd, size_d * sizeof(float)));

    // copy matrices from host to device
    CHECK_HIP_ERROR(hipMemcpy(da, ha.data(), sizeof(float) * size_a, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(db, hb.data(), sizeof(float) * size_b, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dc, hc.data(), sizeof(float) * size_c, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dd, hd.data(), sizeof(float) * size_d, hipMemcpyHostToDevice));

    rocblas_handle handle;
    CHECK_ROCBLAS_ERROR(rocblas_create_handle(&handle));

    CHECK_ROCBLAS_ERROR(rocblas_gemm_ex(handle,
                                        trans_a,
                                        trans_b,
                                        m,
                                        n,
                                        k,
                                        &alpha,
                                        da,
                                        a_type,
                                        lda,
                                        db,
                                        b_type,
                                        ldb,
                                        &beta,
                                        dc,
                                        c_type,
                                        ldc,
                                        dd,
                                        d_type,
                                        ldd,
                                        compute_type,
                                        algo,
                                        solution_index,
                                        flags));

    // copy output from device to CPU
    CHECK_HIP_ERROR(hipMemcpy(hd.data(), dd, sizeof(float) * size_d, hipMemcpyDeviceToHost));

    // calculate golden or correct result
    mat_mat_mult(alpha,
                 beta,
                 m,
                 n,
                 k,
                 ha,
                 a_stride_1,
                 a_stride_2,
                 hb,
                 b_stride_1,
                 b_stride_2,
                 hc,
                 1,
                 ldc,
                 hd_gold,
                 1,
                 ldd);

    if(verbose)
    {
        print_matrix<float>("hd_gold calculated", hd_gold, m, n, ldd);
        print_matrix<float>("hd calculated", hd, m, n, ldd);
    }

    // verify error == 0
    float max_error = 0;
    for(int i_m = 0; i_m < m; i_m++)
    {
        for(int i_n = 0; i_n < n; i_n++)
        {
            float error = hd_gold[i_m + i_n * ldd] - hd[i_m + i_n * ldd];

            error = error >= 0 ? error : -error;

            if(error != 0.0)
            {
                std::cout << i_m << ", " << i_n << ", " << hd_gold[i_m + i_n * ldd] << ", " << hd[i_m + i_n * ldd] << std::endl;
            }

            max_error = error > max_error || isnan(error) ? error : max_error;
        }
    }
    if(max_error > 0 || isnan(max_error))
    {
        std::cout << "FAIL, max_error = " << max_error << std::endl;
    }
    else
    {
        std::cout << "PASS, max_error = " << max_error << std::endl;
    }

    CHECK_HIP_ERROR(hipFree(da));
    CHECK_HIP_ERROR(hipFree(db));
    CHECK_HIP_ERROR(hipFree(dc));
    CHECK_HIP_ERROR(hipFree(dd));
    CHECK_ROCBLAS_ERROR(rocblas_destroy_handle(handle));
    return EXIT_SUCCESS;
}
