
#include <complex>
#include <iostream>
#include <string>
#include <random>

#include <hip/hip_runtime.h>
#include <omp.h>

#include "rocblas.h"
#include "astaux.h"

#define HIP_CHECK(status)                                                                \
    if (status != hipSuccess) {                                                          \
        std::cout << "Got Status: " << status << " at Line: " << __LINE__ << std::endl;  \
        exit(0);                                                                         \
    }

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
void printMatrix_batched(const char* name, T* A, rocblas_int m, rocblas_int n, rocblas_int lda, rocblas_int stride_a, rocblas_int batch_count)
{
    int m_max = 4, n_max = 4, batch_count_max = 4;
    printf("---------- %s ----------\n", name);
    for( int b = 0; b < batch_count && b < batch_count_max; b++)
    {
        for( int i = 0; i < m && i < m_max; i++)
        {
            for( int j = 0; j < n && j < n_max; j++)
            {
                printf("%f ",A[i + j * lda + b * stride_a]);
            }
            printf("\n");
        }
        printf("-----------------------------------------\n");
    }
}

void usage(char *argv[])
{
    std::cerr << "Usage: " << argv[0] << " <options>\n"
        << "options:\n"
        << "\t-h, --help\t\t\t\tShow this help message\n"
        << "\t-p \t\t\tp\t\tprecision s, d, c, z, h\n"
        << "\t-m \t\t\tm\t\trocblas_gemm_ex argument m\n"
        << "\t-n \t\t\tn\t\trocblas_gemm_ex argument n\n"
        << "\t-k \t\t\tk\t\trocblas_gemm_ex argument k\n"
        << "\t--trans_a \t\ttrans_a \tn, N, t, or T\n"
        << "\t--trans_b \t\ttrans_b \tn, N, t, or T\n"
        << "\t--lda \t\t\tlda \t\trocblas_gemm_ex argument lda\n"
        << "\t--ldb \t\t\tldb \t\trocblas_gemm_ex argument ldb\n"
        << "\t--ldc \t\t\tldc \t\trocblas_gemm_ex argument ldc\n"
        << "\t--alpha \t\talpha \t\trocblas_gemm_ex argument alpha\n"
        << "\t--beta \t\t\tbeta \t\trocblas_gemm_ex argument beta\n"
        << "\t--header \t\theader \t\tprint header for output\n"
        << "\t-v, --verbose\t\t\t\tverbose output\n"
        << std::endl;
}


int parse_args(int argc, char *argv[], int &m, int &n, int &k, int &batch_count, int &lda, int &ldb, int &ldc,
                rocblas_operation &trans_a, rocblas_operation &trans_b, 
                float &alpha, float &beta, bool &verbose, char &precision)
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
                else if((arg == "--batch_count") && (i + 1 < argc))
                {
                    batch_count = atoi(argv[++i]);
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
                else if((arg == "--alpha") && (i + 1 < argc))
                {
                    alpha = static_cast<float>(atoi(argv[++i]));
                }
                else if((arg == "--beta") && (i + 1 < argc))
                {
                    beta = static_cast<float>(atoi(argv[++i]));
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





            }
        }
    }
    return EXIT_SUCCESS;
}

template <typename T>
void gemm_ref(rocblas_operation trans_a, rocblas_operation trans_b, rocblas_int m, rocblas_int n, rocblas_int k,
        T alpha, T* a, rocblas_int lda, T* b, rocblas_int ldb, T beta, T* c, rocblas_int ldc)
{
    size_t a_inc_1 = trans_a == rocblas_operation_none ? 1 : lda;
    size_t a_inc_2 = trans_a == rocblas_operation_none ? lda : 1;
    size_t b_inc_1 = trans_b == rocblas_operation_none ? 1 : ldb;
    size_t b_inc_2 = trans_b == rocblas_operation_none ? ldb : 1;
    size_t c_inc_1 = 1, c_inc_2 = ldc;
    for(int i1 = 0; i1 < m; i1++)
    {
        for(int i2 = 0; i2 < n; i2++)
        {
            T temp = 0;
            for(int i3 = 0; i3 < k; i3++)
            {
                temp += a[i1*a_inc_1 + i3* a_inc_2] * b[i3*b_inc_1 + i2*b_inc_2];
            }
            c[i1*c_inc_1 + i2*c_inc_2] = beta * c[i1*c_inc_1 + i2*c_inc_2] + alpha * temp;
        }
    }
}

template <typename T>
void batch_diff(rocblas_int m, rocblas_int n, 
        T* Cref, rocblas_int ldc_ref, rocblas_stride stridec_ref,
        T* h_C, rocblas_int ldc, rocblas_stride stridec,
        rocblas_int batch_count)
{
    for(int i3 = 0; i3 < batch_count; i3++)
    {
        for(int i1 = 0; i1 < m; i1++)
        {
            for(int i2 = 0; i2 < n; i2++)
            {
                if(Cref[i1 + 12 * ldc_ref + i3 * stridec_ref] != h_C[i1 + 12 * ldc + i3 * stridec])
                {
                    std::cout << "ERROR: i1,i2,i3 = " << i1 << ", " << i2 << ", " << i3 << std::endl;
                    return;
                }

            }
        }
    }
    std::cout << "PASS: batch_diff" << std::endl;
}

template <typename T>
void alternating_signs(rocblas_int n1, rocblas_int n2, rocblas_int batch_count, rocblas_int lda, rocblas_int stride_a, T** matrix)
{
    // make b matrix entries have alternating signs like checkerboard
    T sign;
    for (int i3 = 0; i3 < batch_count; i3++)
    {
        for (int i1 = 0; i1 < n1; i1++)
        {
            for (int i2 = 0; i2 < n2; i2++)
            {
                sign = (i1 + i2) & 1 ? 1 : -1;
                (*matrix)[i1 + i2*lda + i3*stride_a] *= sign;
            }
        }
    }
}


//----------------------------------------------------------------------------
template <typename T>
void test_gemm(rocblas_operation trans_a, rocblas_operation trans_b, 
               int m, int n, int k, int lda, int ldb, int ldc,
               T alpha, T beta, int batch_count, int iterations, int pattern, bool verbose)
{
    rocblas_int a_n_1, a_n_2, b_n_1, b_n_2, c_n_1, c_n_2;
    a_n_1 = trans_a == rocblas_operation_none ? m : k;
    a_n_2 = trans_a == rocblas_operation_none ? k : m;
    b_n_1 = trans_b == rocblas_operation_none ? k : n;
    b_n_2 = trans_b == rocblas_operation_none ? n : k;
    c_n_1 = m;
    c_n_2 = n;

    rocblas_int stride_a = lda * a_n_2;
    rocblas_int stride_b = ldb * b_n_2;
    rocblas_int stride_c = ldc * c_n_2;

    std::size_t size_a = stride_a * batch_count;
    std::size_t size_b = stride_b * batch_count;
    std::size_t size_c = stride_c * batch_count;

    //----------
    // GPU setup
    T* h_A = (T*)malloc(sizeof(T)*size_a);
    T* h_B = (T*)malloc(sizeof(T)*size_b);
    T* h_C = (T*)malloc(sizeof(T)*size_c);
    T* Cref = (T*)malloc(sizeof(T)*size_c);
    assert(h_A != nullptr);
    assert(h_B != nullptr);
    assert(h_C != nullptr);
    assert(Cref != nullptr);


    T* d_A;
    T* d_B;
    T* d_C;
    HIP_CHECK(hipMalloc(&d_A, sizeof(T)*size_a));
    HIP_CHECK(hipMalloc(&d_B, sizeof(T)*size_b));
    HIP_CHECK(hipMalloc(&d_C, sizeof(T)*size_c));
    T** h_Aptr = (T**)malloc(sizeof(T*)*batch_count);
    T** h_Bptr = (T**)malloc(sizeof(T*)*batch_count);
    T** h_Cptr = (T**)malloc(sizeof(T*)*batch_count);
    assert(h_Aptr != nullptr);
    assert(h_Bptr != nullptr);
    assert(h_Cptr != nullptr);
    for (int i = 0; i < batch_count; ++i) {
        h_Aptr[i] = d_A + stride_a * i;
        h_Bptr[i] = d_B + stride_b * i;
        h_Cptr[i] = d_C + stride_c * i;
    }

    T** d_Aptr;
    T** d_Bptr;
    T** d_Cptr;
    HIP_CHECK(hipMalloc(&d_Aptr, sizeof(T*)*batch_count));
    HIP_CHECK(hipMalloc(&d_Bptr, sizeof(T*)*batch_count));
    HIP_CHECK(hipMalloc(&d_Cptr, sizeof(T*)*batch_count));
    HIP_CHECK(hipMemcpy(d_Aptr, h_Aptr, sizeof(T*)*batch_count, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_Bptr, h_Bptr, sizeof(T*)*batch_count, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_Cptr, h_Cptr, sizeof(T*)*batch_count, hipMemcpyHostToDevice));


    std::random_device rd;  //Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
    std::uniform_int_distribution<> dis(1, 4);

    for(int i = 0; i < size_a; i++){h_A[i] = dis(gen);}
    for(int i = 0; i < size_b; i++){h_B[i] = dis(gen);}
    for(int i = 0; i < size_c; i++){h_C[i] = dis(gen);}

    alternating_signs(b_n_1, b_n_2, batch_count, ldb, stride_b, &h_B);

    memcpy(Cref, h_C, sizeof(T)*size_c);

    int iseed[4] = {0, 0, 0, 1};

//  hipblasHandle_t handle;
//  hipblasCreate(&handle);

    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));

//  rocblas_handle handle;
//  CHECK_ROCBLAS_ERROR(rocblas_create_handle(&handle));

    //--------
    // GPU run
    double min_time = std::numeric_limits<double>::infinity();

    for (int i = 0; i < iterations; ++i) {

        HIP_CHECK(hipMemcpy(d_A, h_A, sizeof(T)*size_a, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_B, h_B, sizeof(T)*size_b, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_C, h_C, sizeof(T)*size_c, hipMemcpyHostToDevice));

        double start;
        double elapsed;
        start = omp_get_wtime();

            gemm_batched_solution<T>(m, n, k,
                              alpha, d_Aptr, lda,
                                      d_Bptr, ldb,
                              beta,  d_Cptr, ldc,
                              batch_count,
                              stream);
            hipDeviceSynchronize();
        elapsed = omp_get_wtime() - start;
        if (elapsed < min_time)
            min_time = elapsed;
    }

    HIP_CHECK(hipMemcpy(h_C, d_C, sizeof(T)*size_c, hipMemcpyDeviceToHost));
    if(verbose)printMatrix_batched("C    after", h_C, m, n, ldc, stride_c, batch_count);

    double ops;
    double bytes;
    ops = 2.0*m*n*k*batch_count;
    if (std::is_same<T, std::complex<float>>::value ||
        std::is_same<T, std::complex<double>>::value  )
    {
        ops *= 4;
    }
//  printf("\t%lf\n", ops/min_time/1e9);

    bytes = sizeof(T)*m*n + sizeof(T)*m*k + sizeof(T)*k*n;
    if (beta != T(0.0))
        bytes += sizeof(T)*m*n;
    bytes *= batch_count;
//  printf("\t%lf\n", bytes/min_time/1e9);

    //------------------
    // CPU setup and run

    /*
    if(trans_a == rocblas_operation_none)
    {
        printMatrix_batched("h_A        ", h_A, m, k, lda, stride_a, batch_count);
    }
    else
    {
        printMatrix_batched("trans h_A  ", h_A, k, m, lda, stride_a, batch_count);
    }
    printMatrix_batched("h_B        ", h_B, k, n, ldb, stride_b, batch_count);
    printMatrix_batched("Cref before", Cref, m, n, ldc, stride_c, batch_count);
    */

    for (int b = 0; b < batch_count; ++b) {
        gemm_ref(trans_a, trans_b,
                   m, n, k,
                   alpha, &h_A[b*lda*k], lda,
                          &h_B[b*ldb*n], ldb,
                   beta,  &Cref[b*ldc*n], ldc);
    }

    if(verbose) printMatrix_batched("Cref after", Cref, m, n, ldc, stride_c, batch_count);


    // compare GPU (h_C) to CPU (Cref)
    batch_diff(m, n, Cref, ldc, stride_c, h_C, ldc, stride_c, batch_count);


    //--------
    // cleanup
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_Aptr);
    free(h_Bptr);
    free(h_Cptr);
    free(Cref);

    hipFree(d_A);
    hipFree(d_B);
    hipFree(d_C);
    hipFree(d_Aptr);
    hipFree(d_Bptr);
    hipFree(d_Cptr);

    HIP_CHECK(hipStreamDestroy(stream));
}

//----------------------------------------------------------------------------
int main(int argc, char** argv)
{
    int m = 128;
    int n = 128;
    int k = 8;
    int lda = 0;
    int ldb = 0;
    int ldc = 0;
    int batch_count = 2;
    int iterations = 1;
    int pattern = 1;
    rocblas_operation trans_a = rocblas_operation_none;
    rocblas_operation trans_b = rocblas_operation_none;
    bool first = true;
    float alpha = 1, beta = 1;
    char precision = 's';
    bool verbose = false;

    if( parse_args(argc, argv, m, n, k, batch_count, lda, ldb, ldc, trans_a, trans_b, alpha, beta, verbose, precision))
    {
        usage(argv);
        return -1;
    }

    rocblas_int a_n_1 = (trans_a == rocblas_operation_none ? m : k);
    rocblas_int b_n_1 = (trans_b == rocblas_operation_none ? k : n);
    rocblas_int c_n_1 = m;

    lda = lda >= a_n_1 ? lda : a_n_1;
    ldb = ldb >= b_n_1 ? ldb : b_n_1;
    ldc = ldc >= c_n_1 ? ldc : c_n_1;

    char trans_a_char = trans_a == rocblas_operation_none ? 'N' : 'T';
    char trans_b_char = trans_b == rocblas_operation_none ? 'N' : 'T';

    std::cout << "precision, trans_a, trans_b, m, n, k, lda, ldb, ldc, alpha, beta, batch_count = " 
        << precision << trans_a_char << trans_b_char << ", " 
        << m << ", " << n << ", " << k << ", " 
        << lda << ", " << ldb << ", " << ldc << ", " 
        << alpha << ", " << beta << ", " 
        << batch_count << std::endl;

    if(trans_a != rocblas_operation_none || trans_b != rocblas_operation_none)
    {
        std::cout << "ERROR: only NN transpositon supported" << std::endl;
        return (EXIT_FAILURE);
    }
    if(precision == 's')
    {
        test_gemm<float>(trans_a, trans_b, m, n, k, lda, ldb, ldc, alpha, beta, batch_count, iterations, pattern, verbose);
    }
    else if(precision == 'd')
    {
        test_gemm<double>(trans_a, trans_b, m, n, k, lda, ldb, ldc, alpha, beta, batch_count, iterations, pattern, verbose);
    }
    else
    {
        std::cout << "ERROR: precision not implemented" << std::endl;
    }

    printf("\n");

    return (EXIT_SUCCESS);
}
