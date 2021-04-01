
#include <random>
#include <hip/hip_runtime.h>
#include <omp.h>
#include <iostream>

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
            printf("%f ",A[i1 * s1 + i2 * s2]);
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
            for(int i3 = 0; i3 < k; i3++)
            {
                c[i1 * c_s1 + i2 * c_s2] += a[i1 * a_s1 + i3 * a_s2] * b[i2 * b_s1 + i3 * b_s2];
            }
        }
    }

    return;
}

void iterative_algorithm(rocblas_int n, float* c, rocblas_int s1, rocblas_int s2)
{
    for (int i1 = 0; i1 < n; i1++)
        for (int i2 = 0; i2 < n; i2++)
            c[(i1 * s1) + (i2 * s2)] = 0;

    rocblas_int nb_min = 3;
    rocblas_int nb = nb_min;
//  rocblas_int n_diag_blocks = (n + (nb - 1)) / nb;
    rocblas_int ii, nn;

    rocblas_int n_nb, rem, n_rem, skip, i_start = 0;

    n_nb = n / nb;
    rem = n % nb;
    n_rem = rem == 0 ? 0 : 1;

    std::cout << "n, nb_min, n_nb, n_rem = " << n << ", " << nb_min << ", " << n_nb << ", " << n_rem;
    if(n_rem == 1) std::cout << "  rem = " << rem; std::cout << std::endl;


    for (int i = 0; i < n_nb; i++)
    {
        ii = i * nb;
        nn = nb;
        // diag matrix at c[ii, ii], size of diag matrix is nn
        std::cout << "[(" << ii << ", " << ii << "), " << nn << "], ";
        for (int i1 = 0; i1 < nn; i1++)
            for (int i2 = 0; i2 <= i1; i2++)
                c[(ii+i1)*s1 + (ii+i2)*s2] = 1;
    }
    std::cout << std::endl;
    if(n_rem == 1)
    {
        ii = n_nb * nb;
        nn = n - ii;
        // diag matrix at c[ii, ii], size of diag matrix is nn
        std::cout << "[(" << ii << ", " << ii << "), " << nn << "]" << std::endl;
        for (int i1 = 0; i1 < nn; i1++)
            for (int i2 = 0; i2 <= i1; i2++)
                c[(ii+i1)*s1 + (ii+i2)*s2] = 2;
    }


    i_start = 0;


    rocblas_int iteration = 1;

    for (iteration = 1; iteration < 10; iteration++)
    {
        std::cout << std::endl;

        i_start += nb;
        if(i_start > n) break;
        nb = iteration == 1 ? nb_min : nb * 2;
        skip = nb * 2;
        n_nb = (n - i_start) / skip;
        rem  = (n - i_start) % skip;
        if(rem >= nb)
        {
            rem = 0;
            n_nb += 1;
        }
        n_rem = rem == 0 ? 0 : 1;
    
        std::cout << "i_start, nb, skip, n_nb";
        if(n_rem == 1) std::cout << ", n_rem, rem"; std::cout << " = ";
        std::cout << i_start << ", " << nb << ", " << skip << ", " << n_nb << ", " ;
        if(n_rem == 1) std::cout << ", " << n_rem << ", " << rem; std::cout << std::endl;
    
        for(int i = 0; i < n_nb; i++)
        {
            rocblas_int i1 = i_start + (i * skip);
            rocblas_int i2 = i1 - nb;
            std::cout << "[(" << i1 << ", " << i2 << "), " << nb << "],  ";
            for (int i_i1 = 0; i_i1 < nb; i_i1++)
                for (int i_i2 = 0; i_i2 < nb; i_i2++)
                    c[(i1+i_i1)*s1 + (i2+i_i2)*s2] = iteration*2+1;
        }
        std::cout << std::endl;
    
        if(n_rem == 1)
        {
            rocblas_int i1 = i_start + n_nb * skip;
            rocblas_int i2 = i1 - nb;
            rocblas_int n1 = n - i1;
            rocblas_int n2 = nb;
            std::cout << "[i1,i2], [n1,n2] = " << "[" << i1 << ", " << i2 << "], " << "[" << n1 << ", " << n2 << "]" << std::endl;
            for (int i_i1 = 0; i_i1 < n1; i_i1++)
                for (int i_i2 = 0; i_i2 < n2; i_i2++)
                    c[(i1+i_i1)*s1 + (i2+i_i2)*s2] = iteration*2+2;
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

//  rocblas_operation trans = rocblas_operation_none;
    rocblas_operation trans = rocblas_operation_transpose;
    rocblas_fill uplo = rocblas_fill_upper;

    parse_args(argc, argv, n, k, batch_count, lda, ldb, ldc,
                trans, uplo, 
                alpha, beta, verbose, precision);

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

//  rocblas_int lda = rocblas_operation_none == trans ? n : k;
//  rocblas_int ldb = rocblas_operation_none == trans ? n : k;

    int size_a = lda * a_n2;
    int size_b = ldb * b_n2;
    int size_c = ldc * n;

    float* a_h = (float*)malloc(sizeof(float)*size_a); assert(a_h != nullptr);
    float* b_h = (float*)malloc(sizeof(float)*size_b); assert(b_h != nullptr);
    float* c_h = (float*)malloc(sizeof(float)*size_c); assert(c_h != nullptr);
    float* Cref = (float*)malloc(sizeof(float)*size_c); assert(Cref != nullptr);

    iterative_algorithm(n, c_h, 1, ldc);

    print_matrix_pattern("matrix C", rocblas_operation_none, c_h, n, n, ldc); 

    return 0;



    std::random_device rd;  //Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
    std::uniform_int_distribution<> dis(1, 4);

//  for(int i = 0; i < size_a; i++){a_h[i] = dis(gen);}
//  for(int i = 0; i < size_b; i++){b_h[i] = dis(gen);}
//  for(int i = 0; i < size_c; i++){c_h[i] = dis(gen);}

    rocblas_int a_s1 = 1, a_s2 = lda;
    for(int i1 = 0; i1 < a_n1; i1++)
        for(int i2 = 0; i2 < a_n2; i2++)
            a_h[i1 * a_s1 + i2 * a_s2] = i1 + i2;

    rocblas_int b_s1 = 1, b_s2 = ldb;
    for(int i1 = 0; i1 < b_n1; i1++)
        for(int i2 = 0; i2 < b_n2; i2++)
            b_h[i1 * b_s1 + i2 * b_s2] = i1 + i2;

    for(int i1 = 0; i1 < n; i1++)
        for(int i2 = 0; i2 < n; i2++)
            c_h[i1 + i2*ldc] = i1 + i2;

//void printMatrix(const char* name, rocblas_operation trans, T* A, rocblas_int m, rocblas_int n, rocblas_int lda)
    if(verbose)
    {
        print_matrix("matrix A", rocblas_operation_none, a_h, a_n1, a_n2, lda); 
        print_matrix("matrix B", rocblas_operation_none, b_h, a_n1, a_n2, ldb); 
        print_matrix("matrix C", rocblas_operation_none, c_h, n, n, ldc); 
    }

    syrkx_ref(uplo, trans, n, k, alpha, a_h, lda, b_h, ldb, beta, c_h, ldc);

    if(verbose)
    {
        print_matrix("matrix C", rocblas_operation_none, c_h, n, n, ldc); 
    }

    rocblas_int m = n;
    gemm_reference(trans, trans, m, n, k, alpha,
        a_h, lda,
        b_h, ldb, beta,
        c_h, ldc);

    return 0;
}
