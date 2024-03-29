
#include <random>
#include <hip/hip_runtime.h>
#include <omp.h>
#include <iostream>
#include <math.h>

#include "rocblas.h"

template <typename T>
void printMatrix_batched(const char* name, rocblas_operation trans, T* A, rocblas_int m, rocblas_int n, rocblas_int lda, rocblas_int s3, rocblas_int batch_count)
{
    size_t s1 = trans == rocblas_operation_none ? 1 : lda;
    size_t s2 = trans == rocblas_operation_none ? lda : 1;
    int m_max =12, n_max =12, batch_count_max =12;
    printf("---------- %s ----------\n", name);
    for( int b = 0; b < batch_count && b < batch_count_max; b++)
    {
        for( int i1 = 0; i1 < m && i1 < m_max; i1++)
        {
            for( int i2 = 0; i2 < n && i2 < n_max; i2++)
            {
                printf("%f ",A[i1 * s1 + i2 * s2 + b * s3]);
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
        float *a_h, rocblas_int lda,
        float *b_h, rocblas_int ldb,
        float *c_h, rocblas_int ldc)
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

    for(int i1 = 0; i1 < a_n1; i1++)
    for(int i2 = 0; i2 < a_n2; i2++)
            a_h[i1 + i2 * lda] = i1 + i2;

    for(int i1 = 0; i1 < b_n1; i1++)
    for(int i2 = 0; i2 < b_n2; i2++)
            b_h[i1 + i2 * ldb] = i1 + i2;

    for(int i1 = 0; i1 < n; i1++)
    for(int i2 = 0; i2 < n; i2++)
            c_h[i1 + i2*ldc] = 0;
}

bool verify_result(rocblas_int n, float *c_h, float *c_ref, rocblas_int ldc)
{

    bool pass = false;
    for(int i1 = 0; i1 < n; i1++)
    for(int i2 = 0; i2 < n; i2++)
            if(c_h[i1 + i2*ldc] != c_ref[i1 + i2*ldc]) return false;
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
        float* a, rocblas_int lda, 
        float* b, rocblas_int ldb, float beta, 
        float* c, rocblas_int ldc)
{
    rocblas_int a_s1 = rocblas_operation_none == trans ? 1 : lda;
    rocblas_int a_s2 = rocblas_operation_none == trans ? lda : 1;
    rocblas_int b_s1 = rocblas_operation_none == trans ? 1 : ldb;
    rocblas_int b_s2 = rocblas_operation_none == trans ? ldb : 1;
    rocblas_int c_s1 = 1;
    rocblas_int c_s2 = ldc;

    std::cout << "n, k = " << n << ", " << k << std::endl;

    for (int i1 = 0; i1 < n; i1++)
    {
        rocblas_int i2_start = rocblas_fill_lower == uplo ? 0 : i1;
        rocblas_int i2_end   = rocblas_fill_lower == uplo ? i1+1 : n;
        for (int i2 = i2_start; i2 < i2_end; i2++)
        {
            float t = 0;
            for(int i3 = 0; i3 < k; i3++)
            {
                t += a[i1 * a_s1 + i3 * a_s2] * b[i2 * b_s1 + i3 * b_s2];
            }
//          std::cout << "alpha, beta, t, c = " << alpha << ", " << beta << ", " << t << ", " << c[i1 * c_s1 + i2 * c_s2] << std::endl;
            c[i1 * c_s1 + i2 * c_s2] = beta * c[i1 * c_s1 + i2 * c_s2] + alpha * t;
        }
    }

    return;
}

void diag_block(int value, rocblas_fill uplo, rocblas_int nn, rocblas_int i_diag, float* c, rocblas_int ldc)
{
    std::cout << "[(" << i_diag << ", " << i_diag << "), " << nn << "], ";
    for (int i1 = 0; i1 < nn; i1++)
    {
        int i2_start = rocblas_fill_lower == uplo ? 0 : i1;
        int i2_end   = rocblas_fill_lower == uplo ? i1 : nn - 1;
        for (int i2 = i2_start; i2 <= i2_end; i2++)
        {
            c[(i_diag+i1) + (i_diag+i2)*ldc] = value;
        }
    }
}

void gemm_block(int value, rocblas_fill uplo, rocblas_int i1, rocblas_int i2, rocblas_int n1, rocblas_int n2, 
                float* c, rocblas_int ldc) 
{
    if(n1 == n2)
    {
        std::cout << "[(" << i1 << ", " << i2 << "), " << n1 << "],  ";
    }
    else
    {
        std::cout << "[i1,i2], [n1,n2] = " << "[" << i1 << ", " << i2 << "], " << "[" << n1 << ", " << n2 << "]" << std::endl;
    }
    int c_s1 = rocblas_fill_lower == uplo ? 1 : ldc;
    int c_s2 = rocblas_fill_lower == uplo ? ldc : 1;
    for (int i_i1 = 0; i_i1 < n1; i_i1++)
        for (int i_i2 = 0; i_i2 < n2; i_i2++)
            c[(i1+i_i1)*c_s1 + (i2+i_i2)*c_s2] = value;
}

void iterative_algorithm( rocblas_fill uplo, rocblas_operation trans, 
      rocblas_int n, rocblas_int k, float alpha, 
      float *a, rocblas_int lda,
      float *b, rocblas_int ldb, float beta,
      float* c, rocblas_int ldc)
{

    rocblas_int a_s1 = rocblas_operation_none == trans ? 1 : lda;
    rocblas_int b_s1 = rocblas_operation_none == trans ? 1 : ldb;

    rocblas_int nb_min = 3;
    rocblas_int nb = nb_min;
    rocblas_int i_diag, n_diag;

    rocblas_int n_nb, rem, stride, i_start = 0;

    n_nb = n / nb;  // number of diagonal blocks of size nb
    rem = n % nb;   // size of remainder block when n is not multiple of nb

    // diagonal blocks of size nb
    for (int i_nb = 0; i_nb < n_nb; i_nb++)
    {
        i_diag = i_nb * nb; // diag block at c[i_diag, i_diag], size is nb

        syrkx_ref(uplo, trans, 
        nb, k, alpha, 
        &(a[i_diag * a_s1]), lda, 
        &(b[i_diag * b_s1]), ldb, beta, 
        &(c[i_diag + i_diag * ldc]), ldc);
    }

    // remainder diagonal block of size n_diag < nb
    if(rem != 0)
    {
        i_diag = n_nb * nb; // diag block at c[i_diag, i_diag], size is n_diag
        n_diag = n - i_diag; 

        syrkx_ref(uplo, trans, 
        n_diag, k, alpha, 
        &(a[i_diag * a_s1]), lda, 
        &(b[i_diag * b_s1]), ldb, beta, 
        &(c[i_diag + i_diag * ldc]), ldc);
    }

    rocblas_operation trans_a = rocblas_operation_none == trans ? rocblas_operation_none : rocblas_operation_transpose;
    rocblas_operation trans_b = rocblas_operation_none == trans ?  rocblas_operation_transpose : rocblas_operation_none;

    // calls to gemm with m == n == nb. Start with nb == nb_min, and each iteration of the loop nb doubles, and ]
    // the number of gemm calls halves.
    for (nb = nb_min, i_start = nb_min; i_start < n; i_start += nb, nb *= 2)
    {
        stride = nb * 2;
        n_nb = (n - i_start) / stride;
        rem  = (n - i_start) % stride;
        if(rem >= nb)
        {
            rem = 0;
            n_nb += 1;
        }
    
        rocblas_int c_s1, c_s2;
        // gemm blocks of size nb x nb
        for(int i = 0; i < n_nb; i++)
        {
            rocblas_int i1 = i_start + (i * stride);
            rocblas_int i2 = i1 - nb;

            if(rocblas_fill_lower == uplo)
            {
                c_s1 = 1; c_s2 = ldc;
                gemm_reference(trans_a, trans_b, nb, nb, k, alpha,
                    &(a[i1*a_s1]), lda,
                    &(b[i2*b_s1]), ldb, beta,
                    &(c[i1*c_s1+i2*c_s2]), ldc);
            }
            else
            {
                // The lower triangle of c is the transpose of the upper triangle of c
                // if  C = A B^T gives lower then  C^T = B A^T gives upper. We do not
                // need to swap trans_a and trans_b because trans_a ^ T == trans_b 
                c_s1 = ldc; c_s2 = 1;
                gemm_reference(trans_a, trans_b, nb, nb, k, alpha,
                    &(b[i2*b_s1]), ldb,
                    &(a[i1*a_s1]), lda, beta,
                    &(c[i1*c_s1+i2*c_s2]), ldc);
            }
        }
        std::cout << std::endl;
    
        // remainder gemm block of size n1 x nb where n1 < nb
        if(rem != 0)
        {
            rocblas_int i1 = i_start + n_nb * stride;
            rocblas_int i2 = i1 - nb;
            rocblas_int n1 = n - i1;

            if(rocblas_fill_lower == uplo)
            {
                c_s1 = 1; c_s2 = ldc;
                gemm_reference(trans_a, trans_b, n1, nb, k, alpha,
                    &(a[i1*a_s1]), lda,
                    &(b[i2*b_s1]), ldb, beta,
                    &(c[i1*c_s1+i2*c_s2]), ldc);
            }
            else
            {
                // The lower triangle of c is the transpose of the upper triangle of c
                // if  C = A B^T gives lower then  C^T = B A^T gives upper. We do not
                // need to swap trans_a and trans_b because trans_a ^ T == trans_b 
                c_s1 = ldc; c_s2 = 1;
                gemm_reference(trans_a, trans_b, nb, n1, k, alpha,
                    &(b[i2*b_s1]), ldb,
                    &(a[i1*a_s1]), lda, beta, 
                    &(c[i1*c_s1+i2*c_s2]), ldc);
            }
        }
    }
    return;
}


int main(int argc, char** argv)
{
    int n = 4, k = 2, batch_count = 1, lda = 0, ldb = 0, ldc = 0;
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

    int size_a = lda * a_n2; float*   a_h = (float*)malloc(sizeof(float)*size_a); assert(a_h != nullptr);
    int size_b = ldb * b_n2; float*   b_h = (float*)malloc(sizeof(float)*size_b); assert(b_h != nullptr);
    int size_c = ldc * n;    float*   c_h = (float*)malloc(sizeof(float)*size_c); assert(c_h != nullptr);
                             float* c_ref = (float*)malloc(sizeof(float)*size_c); assert(c_ref != nullptr);

    initialize(trans, n, k, a_h, lda, b_h, ldb, c_h, ldc);

    iterative_algorithm( uplo, trans, n, k, alpha, a_h, lda, b_h, ldb, beta, c_h, ldc);

    if(verbose)
    {
        print_matrix("matrix a_h", rocblas_operation_none, a_h, a_n1, a_n2, lda); 
        print_matrix("matrix b_h", rocblas_operation_none, b_h, a_n1, a_n2, ldb); 
        print_matrix("matrix c_h", rocblas_operation_none, c_h, n, n, ldc); 
    }

    initialize(trans, n, k, a_h, lda, b_h, ldb, c_ref, ldc);

    syrkx_ref(uplo, trans, n, k, alpha, a_h, lda, b_h, ldb, beta, c_ref, ldc);

    if(verbose)
    {
        print_matrix("matrix c_ref", rocblas_operation_none, c_ref, n, n, ldc); 
    }

    if (verify_result(n, c_h, c_ref, ldc) == true) 
    {
        std::cout << "--- PASS ---" << std::endl;
    }
    else
    {
        std::cout << "*** FAIL ***" << std::endl;
    }

    return 0;
}
