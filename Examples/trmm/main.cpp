#include <iostream>
#include <cstdio>
#include <iomanip>
#include <vector>
#include <random>
#include <limits>
#include <cstring>
#include <stdlib.h>
#include <hip/hip_runtime.h>
#include "rocblas.h"
//#include "rocblas-types.h"
#include "dcld.hpp"
#include "trmm_reference.hpp"
#include "trmm_gemm_based.hpp"
#include "trmm_recursive.hpp"
//#include "rocblas_trmm.hpp"

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
                  << "\t--side \t\t\tside \t\t l, r for left, right \n"
                  << "\t--uplo \t\t\tuplo \t\t u, l for upper, lower\n"
                  << "\t--trans \t\t\ttrans \t\t t, n for trans, no-trans\n"
                  << "\t--diag \t\t\tdiag \t\t u, n for unit, non-unit\n"
                  << "\t-m \t\t\tm\t\trocblas_gemm_ex argument m\n"
                  << "\t-n \t\t\tn\t\trocblas_gemm_ex argument n\n"
                  << "\t--lda \t\t\tlda \t\trocblas_gemm_ex argument lda\n"
                  << "\t--ldb \t\t\tldb \t\trocblas_gemm_ex argument ldb\n"
                  << "\t--trans \t\ttrans_a \tn, N, t, or T\n"
                  << "\t--alpha \t\talpha \t\trocblas_gemm_ex argument alpha\n"
                  << "\t--header \t\theader \t\tprint header for output\n"
                  << std::endl;
}

static int parse_arguments(
        int argc,
        char *argv[],
        rocblas_side &side,
        rocblas_fill &uplo,
        rocblas_operation &trans,
        rocblas_diagonal &diag,
        rocblas_int &m, 
        rocblas_int &n, 
        rocblas_int &lda, 
        rocblas_int &ldb, 
        float &alpha,
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
                else if((arg == "--lda") && (i + 1 < argc))
                {
                    lda = atoi(argv[++i]);
                }
                else if((arg == "--ldb") && (i + 1 < argc))
                {
                    ldb = atoi(argv[++i]);
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
                else if((arg == "--trans") && (i + 1 < argc))
                {
                    ++i;
                    if(strncmp(argv[i], "T", 1) == 0 || strncmp(argv[i], "t", 1) == 0)
                    {
                        trans = rocblas_operation_transpose;
                    }
                    else if(strncmp(argv[i], "N", 1) == 0 || strncmp(argv[i], "n", 1) == 0)
                    {
                        trans = rocblas_operation_none;
                    }
                    else
                    {
                        std::cerr << "error with " << arg << std::endl;
                        std::cerr << "do not recognize value " << argv[i];
                        return EXIT_FAILURE;
                    }
                }
                else if((arg == "--diag") && (i + 1 < argc))
                {
                    ++i;
                    if(strncmp(argv[i], "N", 1) == 0 || strncmp(argv[i], "n", 1) == 0)
                    {
                        diag = rocblas_diagonal_non_unit;
                    }
                    else if(strncmp(argv[i], "U", 1) == 0 || strncmp(argv[i], "u", 1) == 0)
                    {
                        diag = rocblas_diagonal_unit;
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

     T temp = 0.0;
    for (int i = 0; i < lda; i++)
    {
        for (int j = 0; j < n; j++)
        {
            if(i < m)
            {
                a[i+j*lda] = dis(gen);
                if ((i+j) % 2 == 0)
                {
                    a[i+j*lda] = -a[i+j*lda];
                }
//              a[i+j*lda] = 1.0;
//              a[i+j*lda] = temp++;
            }
            else
            {
                a[i+j*lda] = std::numeric_limits<T>::signaling_NaN();
            }
        }
    }

}

template <typename T>
void initialize_triangular_matrix(
         std::vector<T>           &a, 
         rocblas_int               m, 
         rocblas_int               n, 
         rocblas_int             lda,
         rocblas_side           side,
         rocblas_fill           uplo,
         rocblas_operation     trans,
         rocblas_diagonal       diag)
{
    std::random_device rd;  //Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<T> dis(0.0, 1.0);
    rocblas_int ka = (side == rocblas_side_left) ? m : n;

    // initialize leading ka x ka block
    for (int i = 0; i < ka; i++)
    {
        for (int j = 0; j < ka; j++)
        {
            if(((j <= i) && (uplo == rocblas_fill_lower)) || ((j >= i) && (uplo == rocblas_fill_upper)))
            {
//              a[i+j*lda] = i + j * 100;
                a[i+j*lda] = dis(gen);
//              a[i+j*lda] = 1.0;
            }
            else
            {
                a[i+j*lda] = std::numeric_limits<T>::signaling_NaN();
//              a[i+j*lda] = 2.0;
            }
        }
    }

    // initialize trailing block
    for (int i = ka; i < lda; i++)
    {
        for (int j = 0; j < ka; j++)
        {
            a[i+j*lda] = std::numeric_limits<T>::signaling_NaN();
//          a[i+j*lda] = 2.0;
        }
    }
}

template <typename T>
void template_trmm(rocblas_side side, 
                   rocblas_fill uplo, 
                   rocblas_operation trans,
                   rocblas_diagonal diag,
                   rocblas_int m, 
                   rocblas_int n, 
                   T alpha, 
                   rocblas_int lda, 
                   rocblas_int ldb, 
                   bool verbose)
{

    rocblas_int ka = (side == rocblas_side_left) ? m : n;

    rocblas_int size_a = lda * ka;
    rocblas_int size_b = ldb * n;

    std::vector<T>ha(size_a);
    std::vector<T>hb(size_b);
    std::vector<T>hb_legacy(size_b);
    std::vector<T>hb_gemm_based(size_b);
    std::vector<T>hb_rocblas(size_b);

    initialize_triangular_matrix(ha, m, n, lda, side, uplo, trans, diag);
    initialize_matrix(hb, m, n, ldb);

    hb_legacy = hb;
    hb_gemm_based = hb_legacy;
    hb_rocblas = hb_legacy;

    if(verbose)
    {
        if(side == rocblas_side_left)
        {
            print_matrix("a side_left", ha, lda, m, lda);
        }
        else
        {
            print_matrix("a side_right", ha, lda, n, lda);
        }

        print_matrix("b", hb, ldb, n, ldb);
    }

    rocblas_status status;

      status = trmm_reference( side, uplo, trans, diag, 
              m, n, alpha,
              ha.data(), lda,
              hb_legacy.data(), ldb);

      status = trmm_gemm_based_reference( side, uplo, trans, diag, 
              m, n, alpha,
              ha.data(), lda,
              hb_gemm_based.data(), ldb);

//    status = rocblas_trmm(side, uplo, trans, diag,
//            m, n, alpha,
//            ha.data(), lda,
//            hb_rocblas.data(), ldb);

      status = trmm_recursive(side, uplo, trans, diag,
              m, n, alpha,
              ha.data(), lda,
              hb_rocblas.data(), ldb);


    if(verbose)
    {
        print_matrix("b_legacy output", hb_legacy, ldb, n, ldb);
        print_matrix("b_gemm_based output", hb_gemm_based, ldb, n, ldb);
//      print_matrix("rocblas_trmm", hb_rocblas, ldb, n, ldb);
    }

    T norm_err = 0.0;
    T norm_err_rocblas_trmm = 0.0;
    T norm_ref = 0.0;
    T tolerance = 100;
    T eps = std::numeric_limits<T>::epsilon();
    for (int i1 = 0; i1 < m; i1++)
    {
        for (int i2 = 0; i2 < n; i2++)
        {
            T t = hb_gemm_based[i1+i2*ldb] - hb_legacy[i1+i2*ldb];
            norm_err += t * t;
//          t = hb_rocblas[i1+i2*ldb] - hb_legacy[i1+i2*ldb];
//          norm_err_rocblas_trmm += t * t;
	    if(t != t)
	    {
	        std::cout << "i1, i2, t, norm_err_rocblas_trmm = " << i1 << ", " << i2 << ", " 
		    << t << ", " << norm_err_rocblas_trmm << std::endl;
	    }

            norm_ref += hb_legacy[i1+i2*ldb] * hb_legacy[i1+i2*ldb];
        }
    }
    norm_err = sqrt(norm_err);
//  norm_err_rocblas_trmm = sqrt(norm_err_rocblas_trmm);
    norm_ref = sqrt(norm_ref);
    if (norm_err < norm_ref * eps * tolerance)
    {
        std::cout << "PASS,";
    }
    else
    {
        std::cout << "FAIL, norm_ref * eps * tol = " << norm_ref * eps * tolerance << std::endl;
        std::cout << "  norm_err = " << norm_err;
    }
//  if (norm_err_rocblas_trmm < norm_ref * eps * tolerance)
//  {
//      std::cout << "PASS ";
//  }
//  else
//  {
//      std::cout << "  FAIL, norm_ref * eps * tol = " << norm_ref * eps * tolerance << std::endl;
//      std::cout << "  norm_err = " << norm_err_rocblas_trmm;
//  }
    
//  std::cout << "norm_err = " << norm_err << ", " << norm_err_rocblas_trmm;
}

void strmm(rocblas_side side, 
        rocblas_fill uplo, 
        rocblas_operation trans,
        rocblas_diagonal diag,
        rocblas_int m, 
        rocblas_int n, 
        float alpha, 
        rocblas_int lda, 
        rocblas_int ldb, 
        bool verbose)
{
    template_trmm( side, uplo, trans, diag, m, n, alpha, lda, ldb, verbose);
}

void dtrmm(rocblas_side side, 
        rocblas_fill uplo, 
        rocblas_operation trans,
        rocblas_diagonal diag,
        rocblas_int m, 
        rocblas_int n, 
        float alpha_in, 
        rocblas_int lda, 
        rocblas_int ldb, 
        bool verbose)
{
    double alpha = static_cast<double>(alpha_in);
    template_trmm( side, uplo, trans, diag, m, n, alpha, lda, ldb, verbose);
}


int main(int argc, char* argv[])
{
    rocblas_side side = rocblas_side_left;
    rocblas_fill uplo = rocblas_fill_upper;
    rocblas_operation trans = rocblas_operation_none;
    rocblas_diagonal diag = rocblas_diagonal_non_unit;

    float alpha = 1.0;

    rocblas_int m = 4, n = 3;
    rocblas_int lda = 0, ldb = 0;

    bool verbose = false;
    char precision = 's';

    if(parse_arguments(argc, argv, side, uplo, trans, diag, m, n, lda, ldb, alpha, verbose, precision))
    {
        show_usage(argv);
        return EXIT_FAILURE;
    }

    ldb = ldb < m ? m : ldb;
    if(side == rocblas_side_left)
    {
        lda = lda < m ? m : lda;
    }
    else
    {
        lda = lda < n ? n : lda;
    }

    if(precision == 's' || precision == 'S')
    {
        strmm( side, uplo, trans, diag, m, n, alpha, lda, ldb, verbose);
        std::cout << "  float, ";
    }
    else if(precision == 'd' || precision == 'D')
    {
        dtrmm( side, uplo, trans, diag, m, n, alpha, lda, ldb, verbose);
        std::cout << "  double, ";
    }
//  else if(precision == 'c' || precision == 'C')
//  {
//      ctrmm( side, uplo, trans, diag, m, n, alpha, lda, ldb);
//  }
//  else if(precision == 'z' || precision == 'Z')
//  {
//      ztrmm( side, uplo, trans, diag, m, n, alpha, lda, ldb);
//  }

    std::cout << " m,n,lda,ldb,alpha = " << m << ", " << n << ", " << lda << ", " << ldb << 
    ", " << alpha << ", "; 

    side == rocblas_side_left ? std::cout << "left" : std::cout << "right"; std::cout << ", ";
    uplo == rocblas_fill_upper ? std::cout << "upper" : std::cout << "lower"; std::cout << ", ";
    trans == rocblas_operation_none ? std::cout << "non-transpose" : std::cout << "transpose"; std::cout << ", ";
    diag == rocblas_diagonal_unit ? std::cout << "unit_dagonal" : std::cout << "non_unit_dagonal"; std::cout << std::endl;

    return 0;
}
