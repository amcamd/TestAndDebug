
#include <iostream>
#include <cstdio>
#include <iomanip>
#include <vector>
#include <random>
#include <limits>
#include <cstring>
#include "rocblas-types.h"
#include "symm_reference.hpp"
#include "symm_l3_reference.hpp"

static void show_usage(char* argv[])
{
        std::cerr << "Usage: " << argv[0] << " <options>\n"
                  << "options:\n"
                  << "\t-h, --help\t\t\t\tShow this help message\n"
                  << "\t-v, --verbose\t\t\t\tverbose output\n"
                  << "\t-p \t\t\tp\t\tprecision s, d, c, z, h\n"
                  << "\t--side \t\t\tside \t\t l, r for left, right \n"
                  << "\t--uplo \t\t\tuplo \t\t u, l for upper, lower\n"
                  << "\t-m \t\t\tm\t\trocblas_gemm_ex argument m\n"
                  << "\t-n \t\t\tn\t\trocblas_gemm_ex argument n\n"
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


static int parse_arguments(
        int argc,
        char *argv[],
        rocblas_side &side,
        rocblas_fill &uplo,
        rocblas_int &m, 
        rocblas_int &n, 
        rocblas_int &lda, 
        rocblas_int &ldb, 
        rocblas_int &ldc,
        float &alpha,
        float &beta,
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
                else if((arg == "-m") && (i + 1 < argc))
                {
                    m = atoi(argv[++i]);
                }
                else if((arg == "-n") && (i + 1 < argc))
                {
                    n = atoi(argv[++i]);
                }
                else if((arg == "--alpha") && (i + 1 < argc))
                {
                    alpha = static_cast<float>(atoi(argv[++i]));
                }
                else if((arg == "--beta") && (i + 1 < argc))
                {
                    beta = static_cast<float>(atoi(argv[++i]));
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
                else if((arg == "--side") && (i + 1 < argc))
                {
                    ++i;
                    if(strncmp(argv[i], "L", 1) == 0 || strncmp(argv[i], "l", 1) == 0)
                    {
                        side = rocblas_side_left;
                    }
                    else if(strncmp(argv[i], "R", 1) == 0 || strncmp(argv[i], "r", 1) == 0)
                    {
                        side = rocblas_side_right;
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
                    if(strncmp(argv[i], "L", 1) == 0 || strncmp(argv[i], "l", 1) == 0)
                    {
                        uplo = rocblas_fill_lower;
                    }
                    else if(strncmp(argv[i], "U", 1) == 0 || strncmp(argv[i], "u", 1) == 0)
                    {
                        uplo = rocblas_fill_upper;
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
void print_matrix(
            const char* name, std::vector<T>& A, rocblas_int m, rocblas_int n, rocblas_int lda)
{
    printf("---------- %s ----------\n", name);
    int max_size = 6;
    for(int i = 0; i < m && i < max_size; i++)
    {
        for(int j = 0; j < n && j < max_size; j++)
        {
            std::cout << std::setw(4) << float(A[i + j * lda]) << " ";
        }
        std::cout << "\n";
    }
}


template <typename T>
void initialize_matrix(
        std::vector<T>&     a, 
        rocblas_int         m, 
        rocblas_int         n, 
        rocblas_int       lda)
{
     std::random_device rd;  //Will be used to obtain a seed for the random number engine
     std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
     std::uniform_real_distribution<T> dis(0.0, 1.0);

    for (int i = 0; i < lda; i++)
    {
        for (int j = 0; j < n; j++)
        {
            if(i < m)
            {
//              a[i+j*lda] = dis(gen);
                a[i+j*lda] = 1.0;
            }
            else
            {
                a[i+j*lda] = std::numeric_limits<T>::signaling_NaN();
//              a[i+j*lda] = 2.0;
            }
        }
    }

}

template <typename T>
void initialize_symmetric_matrix(
        std::vector<T>&     a, 
        rocblas_int         m, 
        rocblas_int         n, 
        rocblas_int       lda,
        rocblas_side     side,
        rocblas_fill     uplo)
{
     std::random_device rd;  //Will be used to obtain a seed for the random number engine
     std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
     std::uniform_real_distribution<T> dis(0.0, 1.0);
    rocblas_int ka = (side == rocblas_side_left) ? m : n;

    for (int i = 0; i < lda; i++)
    {
        for (int j = 0; j < ka; j++)
        {
            if((i < ka) && (((j <= i) && (uplo == rocblas_fill_lower)) || ((j >= i) && (uplo == rocblas_fill_upper))))
            {
//              a[i+j*lda] = dis(gen);
                a[i+j*lda] = 1.0;
            }
            else
            {
                a[i+j*lda] = std::numeric_limits<T>::signaling_NaN();
//              a[i+j*lda] = 2.0;
            }
        }
    }

}

template <typename T>
void template_symm(rocblas_side side, 
                   rocblas_fill uplo, 
                   rocblas_int m, 
                   rocblas_int n, 
                   T alpha, 
                   rocblas_int lda, 
                   rocblas_int ldb, 
                   T beta, 
                   rocblas_int ldc)
{
    rocblas_int ka = (side == rocblas_side_left) ? m : n;

    std::vector<T>a(lda*ka);
    std::vector<T>b(ldb*n);
    std::vector<T>c(ldc*n);

    initialize_symmetric_matrix(a, m, n, lda, side, uplo);
    initialize_matrix(b, m, n, ldb);
    initialize_matrix(c, m, n, ldc);
    
    if(side == rocblas_side_left)
    {
        print_matrix("a side_left", a, lda, m, lda);
    }
    else
    {
        print_matrix("a side_right", a, lda, n, lda);
    }

    print_matrix("b", b, ldb, n, ldb);
    print_matrix("c", c, ldc, n, ldc);

    rocblas_status status;

/*
    status = symm_reference( side, uplo, m, n, alpha,
        a.data(), lda,
        b.data(), ldb, beta,
        c.data(), ldc);
*/

  
    status = symm_l3_reference( side, uplo, m, n, alpha,
        a.data(), lda,
        b.data(), ldb, beta,
        c.data(), ldc);
  
    print_matrix("output c", c, ldc, n, ldc);
}

void ssymm(rocblas_side side, 
        rocblas_fill uplo, 
        rocblas_int m, 
        rocblas_int n, 
        float alpha, 
        rocblas_int lda, 
        rocblas_int ldb, 
        float beta, 
        rocblas_int ldc)
{
    template_symm( side, uplo, m, n, alpha, lda, ldb, beta, ldc);
}

void dsymm(rocblas_side side, 
        rocblas_fill uplo, 
        rocblas_int m, 
        rocblas_int n, 
        float alpha_in, 
        rocblas_int lda, 
        rocblas_int ldb, 
        float beta_in, 
        rocblas_int ldc)
{
    double alpha = static_cast<double>(alpha_in);
    double beta = static_cast<double>(beta_in);
    template_symm( side, uplo, m, n, alpha, lda, ldb, beta, ldc);
}


int main(int argc, char* argv[])
{

    int nn = 10, incx = 1, incy = 1;

    float * xx = new float[nn * incx];
    float * yy = new float[nn * incy];

    for (int i = 0; i < nn; i++)
    {
        xx[i] = i;
        yy[i] = -i;
    }


    std::cout << std::endl << "in vector yy" << std::endl;
    for (int i = 0; i < nn; i++)
    {
        std::cout << yy[i] << ", ";
    }
    std::cout << std::endl;

    copy_reference(nn, &xx[0], incx, &yy[0], incy);

    std::cout << std::endl << "out vector yy" << std::endl;
    for (int i = 0; i < nn; i++)
    {
        std::cout << yy[i] << ", ";
    }
    std::cout << std::endl;

//  return 0;



    rocblas_side side = rocblas_side_right;
    rocblas_fill uplo = rocblas_fill_lower;

    float alpha = 1.0;
    float beta = 2.0;

    rocblas_int m = 4, n = 3;
    rocblas_int lda = 0, ldb = 0, ldc = 0;

    bool verbose = false;
    char precision = 's';

    if(parse_arguments(argc, argv, side, uplo, m, n, lda, ldb, ldc, alpha, beta, verbose, precision))
    {
        show_usage(argv);
        return EXIT_FAILURE;
    }

    ldb = ldb < m ? m : ldb;
    ldc = ldc < m ? m : ldc;
    if(side == rocblas_side_left)
    {
        lda = lda < m ? m : lda;
    }
    else
    {
        lda = lda < n ? n : lda;
    }

    std::cout << "m,n,lda,ldb,ldc,alpha,beta = " << m << ", " << n << ", " << lda << ", " << ldb << 
    ", " << ldc << ", " << alpha << ", " << beta << ", "; 

    side == rocblas_side_left ? std::cout << "left" : std::cout << "right"; std::cout << ", ";
    uplo == rocblas_fill_upper ? std::cout << "upper" : std::cout << "lower"; std::cout << ", ";

    if(precision == 's' || precision == 'S')
    {
        std::cout << "float" << std::endl;
        ssymm( side, uplo, m, n, alpha, lda, ldb, beta, ldc);
    }
    else if(precision == 'd' || precision == 'D')
    {
        std::cout << "double" << std::endl;
        dsymm( side, uplo, m, n, alpha, lda, ldb, beta, ldc);
    }
//  else if(precision == 'c' || precision == 'C')
//  {
//      csymm( side, uplo, m, n, alpha, lda, ldb, beta, ldc);
//  }
//  else if(precision == 'z' || precision == 'Z')
//  {
//      zsymm( side, uplo, m, n, alpha, lda, ldb, beta, ldc);
//  }

    return 0;
}
