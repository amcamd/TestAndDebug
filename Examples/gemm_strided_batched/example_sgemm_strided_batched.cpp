#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <limits>
#include <iostream>
#include <cstring>

// default sizes
#define DIM1 127
#define DIM2 128
#define DIM3 129
#define BATCH_COUNT 10
#define ALPHA 2
#define BETA 3


using namespace std;

static int parse_arguments(int argc,
                           char* argv[],
                           int& m,
                           int& n,
                           int& k,
                           int& lda,
                           int& ldb,
                           int& ldc,
                           int& stride_a,
                           int& stride_b,
                           int& stride_c,
                           int& batch_count,
                           float& alpha,
                           float& beta,
                           char& trans_a,
                           char& trans_b,
                           bool& header,
                           bool& name,
                           bool& verbose,
                           bool& gemm,
                           bool& gemm_strided_batched,
                           char& precision
                           )
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
                else if((arg == "-f") && (i + 1 < argc))
                {
                    ++i;
                    if(strncmp(argv[i], "gemm_strided_batched", 20) == 0)
                    {
                        gemm_strided_batched = true;
                    }
                    else if(strncmp(argv[i], "gemm", 4) == 0)
                    {
                        gemm = true;
                    } 
                }
                else if((arg == "-r") && (i + 1 < argc))
                {
                    precision = *(argv[++i]);
                }
                else if(arg == "--header")
                {
                    header = true;
                }
                else if(arg == "--name")
                {
                    name = true;
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
                else if((arg == "--batch") && (i + 1 < argc))
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
                else if((arg == "--stride_a") && (i + 1 < argc))
                {
                    stride_a = atoi(argv[++i]);
                }
                else if((arg == "--stride_b") && (i + 1 < argc))
                {
                    stride_b = atoi(argv[++i]);
                }
                else if((arg == "--stride_c") && (i + 1 < argc))
                {
                    stride_c = atoi(argv[++i]);
                }
                else if((arg == "--alpha") && (i + 1 < argc))
                {
                    alpha = atof(argv[++i]);
                }
                else if((arg == "--beta") && (i + 1 < argc))
                {
                    beta = atof(argv[++i]);
                }
                else if((arg == "--transposeA") && (i + 1 < argc))
                {
                    ++i;
                    if(strncmp(argv[i], "N", 1) == 0 || strncmp(argv[i], "n", 1) == 0)
                    {
                        trans_a = 'N';
                    }
                    else if(strncmp(argv[i], "T", 1) == 0 || strncmp(argv[i], "t", 1) == 0)
                    {
                        trans_a = 'T';
                    }
                    else
                    {
                        std::cerr << "error with " << arg << std::endl;
                        std::cerr << "do not recognize value " << argv[i];
                        return EXIT_FAILURE;
                    }
                }
                else if((arg == "--transposeB") && (i + 1 < argc))
                {
                    ++i;
                    if(strncmp(argv[i], "N", 1) == 0 || strncmp(argv[i], "n", 1) == 0)
                    {
                        trans_b = 'N';
                    }
                    else if(strncmp(argv[i], "T", 1) == 0 || strncmp(argv[i], "t", 1) == 0)
                    {
                        trans_b = 'T';
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

bool bad_argument(char trans_a,
                  char trans_b,
                  int m,
                  int n,
                  int k,
                  int lda,
                  int ldb,
                  int ldc,
                  int stride_a,
                  int stride_b,
                  int stride_c,
                  int batch_count)
{
    bool argument_error = false;

    if((trans_a == 'N') && (lda < m))
    {
        argument_error = true;
        std::cerr << "ERROR: bad argument lda = " << lda << " < " << m << std::endl;
    }
    if((trans_a == 'T') && (lda < k))
    {
        argument_error = true;
        std::cerr << "ERROR: bad argument lda = " << lda << " < " << k << std::endl;
    }
    if((trans_b == 'N') && (ldb < k))
    {
        argument_error = true;
        std::cerr << "ERROR: bad argument ldb = " << ldb << " < " << k << std::endl;
    }
    if((trans_b == 'T') && (ldb < n))
    {
        argument_error = true;
        std::cerr << "ERROR: bad argument ldb = " << ldb << " < " << n << std::endl;
    }
    if(stride_a < 0)
    {
        argument_error = true;
        std::cerr << "ERROR: bad argument stride_a < 0" << std::endl;
    }
    if(stride_b < 0)
    {
        argument_error = true;
        std::cerr << "ERROR: bad argument stride_b < 0" << std::endl;
    }
    if(ldc < m)
    {
        argument_error = true;
        std::cerr << "ERROR: bad argument ldc = " << ldc << " < " << m << std::endl;
    }
    if(stride_c < n * ldc)
    {
        argument_error = true;
        std::cerr << "ERROR: bad argument stride_c = " << stride_c << " < " << n * ldc << std::endl;
    }
    if(batch_count < 1)
    {
        argument_error = true;
        std::cerr << "ERROR: bad argument batch_count = " << batch_count << " < 1" << std::endl;
    }

    return argument_error;
}

int main(int argc, char* argv[])
{
    // initialize parameters with default values
    char trans_a = 'N';
    char trans_b = 'T';

    char precision = 's';

    // invalid int and float for rocblas_sgemm_strided_batched int and float arguments
    int invalid_int = std::numeric_limits<int>::min() + 1;
    float invalid_float     = std::numeric_limits<float>::quiet_NaN();

    // initialize to invalid value to detect if values not specified on command line
    int m = invalid_int, lda = invalid_int, stride_a = invalid_int;
    int n = invalid_int, ldb = invalid_int, stride_b = invalid_int;
    int k = invalid_int, ldc = invalid_int, stride_c = invalid_int;

    int batch_count = invalid_int;

    float alpha = invalid_float;
    float beta  = invalid_float;

    bool verbose = false;
    bool header  = false;
    bool name    = false;
    bool gemm = false;
    bool gemm_strided_batched = false;

    if(parse_arguments(argc,
                       argv,
                       m,
                       n,
                       k,
                       lda,
                       ldb,
                       ldc,
                       stride_a,
                       stride_b,
                       stride_c,
                       batch_count,
                       alpha,
                       beta,
                       trans_a,
                       trans_b,
                       header,
                       name,
                       verbose,
                       gemm,
                       gemm_strided_batched,
                       precision))
    {
//      show_usage(argv);
        std::cout << "parsing error" << std::endl;
        return EXIT_FAILURE;
    }

    // when arguments not specified, set to default values
    if(m == invalid_int)
        m = DIM1;
    if(n == invalid_int)
        n = DIM2;
    if(k == invalid_int)
        k = DIM3;
    if(lda == invalid_int)
        lda = trans_a == 'N' ? m : k;
    if(ldb == invalid_int)
        ldb = trans_b == 'N' ? k : n;
    if(ldc == invalid_int)
        ldc = m;
    if(stride_a == invalid_int)
        stride_a = trans_a == 'N' ? lda * k : lda * m;
    if(stride_b == invalid_int)
        stride_b = trans_b == 'N' ? ldb * n : ldb * k;
    if(stride_c == invalid_int)
        stride_c = ldc * n;
    if(alpha != alpha)
        alpha = ALPHA; // check for alpha == invalid_float == NaN
    if(beta != beta)
        beta = BETA; // check for beta == invalid_float == NaN
    if(batch_count == invalid_int)
        batch_count = BATCH_COUNT;

    if(bad_argument(
           trans_a, trans_b, m, n, k, lda, ldb, ldc, stride_a, stride_b, stride_c, batch_count))
    {
//      show_usage(argv);
        std::cout << "bad argument" << std::endl;
        return EXIT_FAILURE;
    }

    if(header)
    {
        std::cout << "transAB,M,N,K,lda,ldb,ldc,stride_a,stride_b,stride_c,batch_count,alpha,beta,"
                     "result,error";
        if(name)
        {
            std::cout << ",name";
        }
        std::cout << std::endl;
    }

    int a_stride_1, a_stride_2, b_stride_1, b_stride_2;
    int size_a1, size_b1, size_c1 = ldc * n;
    if(trans_a == 'N')
    {
        cout << 'N';
        a_stride_1 = 1;
        a_stride_2 = lda;
        size_a1    = lda * k;
    }
    else
    {
        cout << 'T';
        a_stride_1 = lda;
        a_stride_2 = 1;
        size_a1    = lda * m;
    }
    if(trans_b == 'N')
    {
        cout << "N, ";
        b_stride_1 = 1;
        b_stride_2 = ldb;
        size_b1    = ldb * n;
    }
    else
    {
        cout << "T, ";
        b_stride_1 = ldb;
        b_stride_2 = 1;
        size_b1    = ldb * k;
    }

    std::cout << std::endl;
    std::cout << "gemm                 = " << gemm << std::endl;
    std::cout << "gemm_strided_batched = " << gemm_strided_batched << std::endl;
    if(gemm_strided_batched)
    {
        std::cout << "gemm_strided_gatched_tuple  db_sb{ {"
                  << m << ", " << n << ", " << k << ", " << lda << ", " << ldb << ", " << ldc << ", "
                  << stride_a << ", " << stride_b << ", " << stride_c << "}, "
                  << "{" << alpha << ", " << beta << "}, "
                  << "{'" << trans_a << "', '" << trans_b << "'}, "
                  << batch_count << "};"
                  << std::endl;
    }
    else if (gemm)
    {
        std::cout << "gemm_tuple  db_sb{ {"
                  << m << ", " << n << ", " << k << ", " << lda << ", " << ldb << ", " << ldc << "}, "
                  << "{" << alpha << ", " << beta << "}, "
                  << "{'" << trans_a << "', '" << trans_b << "'}};"
                  << std::endl;
    }
}
