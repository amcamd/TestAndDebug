
#include <random>
#include <hip/hip_runtime.h>
#include <omp.h>
#include <iostream>
#include <math.h>

#include "rocblas.h"
#include "gemm_batched.h"

#define HIP_CHECK(status)                                                                \
    if (status != hipSuccess) {                                                          \
        std::cout << "Got Status: " << status << " at Line: " << __LINE__ << std::endl;  \
        exit(0);                                                                         \
    }

template <typename T>
void print_matrix_strided_batched(const char* name, rocblas_operation trans, T* A, rocblas_int m, rocblas_int n, rocblas_int lda, rocblas_int stride, rocblas_int batch_count)
{
    size_t s1 = trans == rocblas_operation_none ? 1 : lda;
    size_t s2 = trans == rocblas_operation_none ? lda : 1;
    int m_max =12, n_max =12, batch_count_max =12;
    printf("---------- %s ----------\n", name);
    for( int i3 = 0; i3 < batch_count && i3 < batch_count_max; i3++)
    {
        for( int i1 = 0; i1 < m && i1 < m_max; i1++)
        {
            for( int i2 = 0; i2 < n && i2 < n_max; i2++)
            {
                printf("%4.0f ",A[i1 * s1 + i2 * s2 + i3 * stride]);
            }
            printf("\n");
        }
        printf("-----------------------------------------\n");
    }
}

template <typename T>
void print_matrix(const char* name, rocblas_operation trans, T* A, rocblas_int m, rocblas_int n, rocblas_int lda)
{
    size_t s1 = trans == rocblas_operation_none ? 1 : lda;
    size_t s2 = trans == rocblas_operation_none ? lda : 1;
    int m_max =12, n_max =12;
    printf("---------- %s ----------\n", name);
    for( int i1 = 0; i1 < m && i1 < m_max; i1++)
    {
        for( int i2 = 0; i2 < n && i2 < n_max; i2++)
        {
            printf("%4.0f",A[i1 * s1 + i2 * s2]);
        }
        printf("\n");
    }
}

template <typename T>
void print_matrix_pattern(const char* name, rocblas_operation trans, T* A, rocblas_int m, rocblas_int n, rocblas_int lda)
{
    size_t s1 = trans == rocblas_operation_none ? 1 : lda;
    size_t s2 = trans == rocblas_operation_none ? lda : 1;
    int m_max =60, n_max =60;
    printf("---------- %s ----------\n", name);
    for( int i1 = 0; i1 < m && i1 < m_max; i1++)
    {
        for( int i2 = 0; i2 < n && i2 < n_max; i2++)
        {
            printf("%2d ",static_cast<int>(A[i1 * s1 + i2 * s2]));
        }
        printf("\n");
    }
}


void usage(char *argv[])
{
    std::cerr << "Usage: " << argv[0] << " <options>\n"
        << "options:\n"
        << "\t-h, --help\t\t\t\tShow this help message\n"
        << "\t-p \t\t\tp\t\tprecision s, d, c, z, h\n"
        << "\t-n \t\t\tn\t\trocblas_gemm_ex argument n\n"
        << "\t-k \t\t\tk\t\trocblas_gemm_ex argument k\n"
        << "\t--batch_count \t\tbatch_count \trocblas_gemm_ex argument batch_count\n"
        << "\t--trans \t\ttrans \tn, N, t, or T\n"
        << "\t--uplo \t\tuplo \tu, U, l, or L\n"
        << "\t--lda \t\t\tlda \t\trocblas_gemm_ex argument lda\n"
        << "\t--ldb \t\t\tldb \t\trocblas_gemm_ex argument ldb\n"
        << "\t--ldc \t\t\tldc \t\trocblas_gemm_ex argument ldc\n"
        << "\t--alpha \t\talpha \t\trocblas_gemm_ex argument alpha\n"
        << "\t--beta \t\t\tbeta \t\trocblas_gemm_ex argument beta\n"
        << "\t-v, --verbose\t\t\t\tverbose output\n"
        << std::endl;
}

int parse_args(int argc, char *argv[], int &n, int &k, int &batch_count, int &lda, int &ldb, int &ldc,
                rocblas_operation &trans, rocblas_fill &uplo, 
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

                else if((arg == "--trans") && (i + 1 < argc))
                {
                    ++i;
                    if(strncmp(argv[i], "N", 1) == 0 || strncmp(argv[i], "n", 1) == 0)
                    {
                        trans = rocblas_operation_none;
                    }
                    else if(strncmp(argv[i], "T", 1) == 0 || strncmp(argv[i], "t", 1) == 0)
                    {
                        trans = rocblas_operation_transpose;
                    }
                    else
                    {
                        std::cerr << "error with " << arg << std::endl;
                        std::cerr << "do not recognize value " << argv[i];
                        return EXIT_FAILURE;
                    }
                }
                else if((arg == "--uplo") && (i + 1 < argc))
                {
                    ++i;
                    if(strncmp(argv[i], "U", 1) == 0 || strncmp(argv[i], "u", 1) == 0)
                    {
                        uplo = rocblas_fill_upper;
                    }
                    else if(strncmp(argv[i], "L", 1) == 0 || strncmp(argv[i], "l", 1) == 0)
                    {
                        uplo = rocblas_fill_lower;
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

void initialize(rocblas_operation trans, rocblas_int n, rocblas_int k, 
        float *a_h, rocblas_int lda, rocblas_int stride_a,
        float *b_h, rocblas_int ldb, rocblas_int stride_b,
        float *c_h, rocblas_int ldc, rocblas_int stride_c, int batch_count)
{
    std::random_device rd;  //Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
    std::uniform_int_distribution<> dis(1, 4);

//  for(int i = 0; i < size_a; i++){a_h[i] = dis(gen);}
//  for(int i = 0; i < size_b; i++){b_h[i] = dis(gen);}
//  for(int i = 0; i < size_c; i++){c_h[i] = dis(gen);}

    rocblas_int a_n1 = rocblas_operation_none == trans ? n : k;
    rocblas_int b_n1 = rocblas_operation_none == trans ? n : k;

    rocblas_int a_n2 = rocblas_operation_none == trans ? k : n;
    rocblas_int b_n2 = rocblas_operation_none == trans ? k : n;

    for(int i3 = 0; i3 < batch_count; i3++)
    for(int i1 = 0; i1 < a_n1; i1++)
    for(int i2 = 0; i2 < a_n2; i2++)
            a_h[i1 + i2 * lda + i3*stride_a] = i1 + i2 + i3;

    for(int i3 = 0; i3 < batch_count; i3++)
    for(int i1 = 0; i1 < b_n1; i1++)
    for(int i2 = 0; i2 < b_n2; i2++)
            b_h[i1 + i2 * ldb + i3*stride_b] = i1 + i2 + i3;

    for(int i3 = 0; i3 < batch_count; i3++)
    for(int i1 = 0; i1 < n; i1++)
    for(int i2 = 0; i2 < n; i2++)
            c_h[i1 + i2*ldc + i3*stride_c] = 0;
}

bool verify_result(rocblas_int n, float *c_h, float *c_ref, rocblas_int ldc, rocblas_int stride_c, rocblas_int batch_count)
{

    bool pass = false;
    for(int i3 = 0; i3 < batch_count; i3++)
    for(int i1 = 0; i1 < n; i1++)
    for(int i2 = 0; i2 < n; i2++)
            if(c_h[i1 + i2*ldc + i3*batch_count] != c_ref[i1 + i2*ldc + i3*batch_count]) return false;
    return true;

}


template <typename T>
void mat_mat_mult(T alpha, T beta, int M, int N, int K,
        T* a, int as1, int as2,
        T* b, int bs1, int bs2,
        T* c, int cs1, int cs2)
{
    for(int i1=0; i1<M; i1++)
    {
        for(int i2=0; i2<N; i2++)
        {
            T t = 0.0;
            for(int i3=0; i3<K; i3++)
            {
                t +=  a[i1 * as1 + i3 * as2] * b[i3 * bs1 + i2 * bs2];
            }
            c[i1*cs1 +i2*cs2] = beta * c[i1*cs1+i2*cs2] + alpha * t ;
        }
    }
}

template <typename T>
rocblas_status gemm_reference(rocblas_operation transA, rocblas_operation transB,
        rocblas_int m, rocblas_int n, rocblas_int k, T alpha,
        T* a, rocblas_int lda,
        T* b, rocblas_int ldb, T beta,
        T* c, rocblas_int ldc)
{
    rocblas_int a_s1 = transA == rocblas_operation_none ? 1 : lda;
    rocblas_int a_s2 = transA == rocblas_operation_none ? lda : 1;

    rocblas_int b_s1 = transB == rocblas_operation_none ? 1 : ldb;
    rocblas_int b_s2 = transB == rocblas_operation_none ? ldb : 1;

    rocblas_int c_s1 = 1, c_s2 = ldc;

    for(int i1 = 0; i1 < m; i1++)
    {
        for(int i2 = 0; i2 < n; i2++)
        {
            T t = 0.0;
            for(int i3 = 0; i3 < k; i3++)
            {
                t +=  a[i1*a_s1 + i3*a_s2] * b[i3*b_s1 + i2*b_s2];
            }
            c[i1*c_s1 + i2*c_s2] = beta * c[i1*c_s1 + i2*c_s2] + alpha * t ;
        }
    }

    return rocblas_status_success;
}


void syrkx_ref(rocblas_fill uplo, rocblas_operation trans, 
        rocblas_int n, rocblas_int k, float alpha, 
        float* a, rocblas_int lda, rocblas_int stride_a, 
        float* b, rocblas_int ldb, rocblas_int stride_b, float beta, 
        float* c, rocblas_int ldc, rocblas_int stride_c, rocblas_int batch_count)
{
    rocblas_int a_s1 = rocblas_operation_none == trans ? 1 : lda;
    rocblas_int a_s2 = rocblas_operation_none == trans ? lda : 1;
    rocblas_int b_s1 = rocblas_operation_none == trans ? 1 : ldb;
    rocblas_int b_s2 = rocblas_operation_none == trans ? ldb : 1;
    rocblas_int c_s1 = 1;
    rocblas_int c_s2 = ldc;

    std::cout << "n, k = " << n << ", " << k << std::endl;

    for (int i4 = 0; i4 < batch_count; i4++)
    {
    for (int i1 = 0; i1 < n; i1++)
    {
        rocblas_int i2_start = rocblas_fill_lower == uplo ? 0 : i1;
        rocblas_int i2_end   = rocblas_fill_lower == uplo ? i1+1 : n;
        for (int i2 = i2_start; i2 < i2_end; i2++)
        {
            float t = 0;
            for(int i3 = 0; i3 < k; i3++)
            {
                t += a[i1 * a_s1 + i3 * a_s2 + i4 * stride_a] * b[i2 * b_s1 + i3 * b_s2 + i4 * stride_b];
            }
//          std::cout << "alpha, beta, t, c = " << alpha << ", " << beta << ", " << t << ", " << c[i1 * c_s1 + i2 * c_s2] << std::endl;
            c[i1 * c_s1 + i2 * c_s2 + i4 * stride_c] = beta * c[i1 * c_s1 + i2 * c_s2 + i4 * stride_c] + alpha * t;
        }
    }
    }

    return;
}


template <typename T>
void syrkx_strided_batched(rocblas_fill uplo, rocblas_operation trans, 
        rocblas_int n, rocblas_int k, T alpha, 
        T* h_a, rocblas_int lda, rocblas_stride stride_a, 
        T* h_b, rocblas_int ldb, rocblas_stride stride_b, T beta, 
        T* h_c, rocblas_int ldc, rocblas_stride stride_c, rocblas_int batch_count)
{
    T* d_a; int size_a = stride_a * batch_count; HIP_CHECK(hipMalloc(&d_a, sizeof(T)*size_a));
    T* d_b; int size_b = stride_b * batch_count; HIP_CHECK(hipMalloc(&d_b, sizeof(T)*size_b));
    T* d_c; int size_c = stride_c * batch_count; HIP_CHECK(hipMalloc(&d_c, sizeof(T)*size_c));

    HIP_CHECK(hipMemcpy(d_a, h_a, sizeof(T)*size_a, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_b, h_b, sizeof(T)*size_b, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_c, h_c, sizeof(T)*size_c, hipMemcpyHostToDevice));

    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));

    // convert to gemm
    rocblas_operation trans_b = rocblas_operation_none == trans ?  rocblas_operation_transpose : rocblas_operation_none;
    rocblas_int m = n;

    syrkx_batched_solution(trans, trans_b, m, n, k, alpha,
                               d_a, lda, stride_a,
                               d_b, ldb, stride_b, beta,
                               d_c, ldc, stride_c, batch_count, stream);





    HIP_CHECK(hipMemcpy(h_c, d_c, sizeof(T)*size_c, hipMemcpyDeviceToHost));

    hipFree(d_a);
    hipFree(d_b);
    hipFree(d_c);

    HIP_CHECK(hipStreamDestroy(stream));
}


int main(int argc, char** argv)
{
    int n = 4, k = 2, batch_count = 2, lda = 0, ldb = 0, ldc = 0;
    float alpha = 1.0, beta = 1.0;
    bool verbose = true;
    char precision = 's';

    rocblas_operation trans = rocblas_operation_none;
    rocblas_fill uplo = rocblas_fill_upper;

    parse_args(argc, argv, n, k, batch_count, lda, ldb, ldc,
                trans, uplo, 
                alpha, beta, verbose, precision);

    std::cout << "uplo, trans, n, k, alpha, beta = ";
    if(rocblas_fill_upper == uplo) { std::cout << "U,";} else { std::cout << "L,";}
    if(rocblas_operation_none == trans) { std::cout << "N,";} else { std::cout << "T, ";} 
    std::cout << n << ", " << k << ", " << alpha << ", " << beta << std::endl;


    rocblas_int a_n1, a_n2, b_n1, b_n2;

    if(rocblas_operation_none == trans)
    {
        a_n1 = n; a_n2 = k; 
        b_n1 = n; b_n2 = k;
    }
    else
    {
        a_n1 = k; a_n2 = n;
        b_n1 = k; b_n2 = n;
    }
    if(lda < a_n1)lda = a_n1;
    if(ldb < b_n1)ldb = b_n1;
    if(ldc < n)ldc = n;

    rocblas_stride stride_a = lda * a_n2; int size_a = stride_a * batch_count;
    rocblas_stride stride_b = ldb * b_n2; int size_b = stride_b * batch_count;
    rocblas_stride stride_c = ldc * n;    int size_c = stride_c * batch_count;

    float*   a_h = (float*)malloc(sizeof(float)*size_a); assert(a_h != nullptr);
    float*   b_h = (float*)malloc(sizeof(float)*size_b); assert(b_h != nullptr);
    float*   c_h = (float*)malloc(sizeof(float)*size_c); assert(c_h != nullptr);
    float* c_ref = (float*)malloc(sizeof(float)*size_c); assert(c_ref != nullptr);

    initialize(trans, n, k, a_h, lda, stride_a, b_h, ldb, stride_b, c_h, ldc, stride_c, batch_count);

    syrkx_strided_batched(uplo, trans, 
        n, k, alpha, 
        a_h, lda, stride_a, 
        b_h, ldb, stride_b, beta, 
        c_h, ldc, stride_c, batch_count);


    if(verbose)
    {
        print_matrix_strided_batched("matrix a_h", rocblas_operation_none, a_h, a_n1, a_n2, lda, stride_a, batch_count); 
        print_matrix_strided_batched("matrix b_h", rocblas_operation_none, b_h, a_n1, a_n2, ldb, stride_b, batch_count); 
        print_matrix_strided_batched("matrix c_h", rocblas_operation_none, c_h, n, n, ldc, stride_c, batch_count); 
    }

    initialize(trans, n, k, a_h, lda, stride_a, b_h, ldb, stride_b, c_ref, ldc, stride_c, batch_count);

    syrkx_ref(uplo, trans, n, k, alpha, a_h,   lda, stride_a, 
                                        b_h,   ldb, stride_b, beta, 
                                        c_ref, ldc, stride_c, batch_count);

    if(verbose)
    {
        print_matrix_strided_batched("matrix c_ref", rocblas_operation_none, c_ref, n, n, ldc, stride_c, batch_count); 
    }

    if (verify_result(n, c_h, c_ref, ldc, stride_c, batch_count) == true) 
    {
        std::cout << "--- PASS ---" << std::endl;
    }
    else
    {
        std::cout << "*** FAIL ***" << std::endl;
    }

    return 0;
}
