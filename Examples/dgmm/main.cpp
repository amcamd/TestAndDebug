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
#include "rocblas-types.h"
#include "dgmm_reference.hpp"
#include "rocblas_dgmm.hpp"

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
                  << "\t-m \t\t\tm\t\trocblas_dgmm argument m\n"
                  << "\t-n \t\t\tn\t\trocblas_dgmm argument n\n"
                  << "\t--lda \t\t\tlda \t\trocblas_dgmm argument lda\n"
                  << "\t--incx \t\t\tincx \t\trocblas_dgmm argument incx\n"
                  << "\t--ldc \t\t\tldc \t\trocblas_dgmm argument ldc\n"
                  << "\t--header \t\theader \t\tprint header for output\n"
                  << std::endl;
}

static int parse_arguments(
        int argc,
        char *argv[],
        rocblas_side &side,
        rocblas_int &m, 
        rocblas_int &n, 
        rocblas_int &lda, 
        rocblas_int &ldc, 
        rocblas_int &incx,
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
                else if((arg == "--lda") && (i + 1 < argc))
                {
                    lda = atoi(argv[++i]);
                }
                else if((arg == "--ldc") && (i + 1 < argc))
                {
                    ldc = atoi(argv[++i]);
                }
                else if((arg == "--incx") && (i + 1 < argc))
                {
                    incx = atoi(argv[++i]);
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
            }
        }
    }
    return EXIT_SUCCESS;
}

template <typename T>
void print_vector(
            const char* name, std::vector<T>& x, rocblas_int n, rocblas_int incx)
{
    printf("---------- %s ----------\n", name);
    int max_n = 10;
    for(int i = 0; i < n && i < max_n; i++)
    {
        std::cout << std::setw(4) << float(x[i * incx]) << " ";
    }
    std::cout << std::endl;
}

template <typename T>
void print_matrix(
            const char* name, std::vector<T>& A, rocblas_int m, rocblas_int n, rocblas_int lda)
{
    printf("---------- %s ----------\n", name);
    int max_i = 10;
    int max_j = 10;
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
                a[i+j*lda] = temp++;
            }
            else
            {
                a[i+j*lda] = std::numeric_limits<T>::signaling_NaN();
            }
        }
    }

}

template <typename T>
void initialize_vector(
        std::vector<T>&     x, 
        rocblas_int         n, 
        rocblas_int       incx)
{
    std::random_device rd;  //Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<T> dis(0.0, 1.0);

    for (int i = 0; i < n; i++)
    {
//      x[i*incx] = dis(gen);
        x[i*incx] = i + 1;
    }
}


template <typename T>
void template_dgmm(rocblas_side side, 
                   rocblas_int m, 
                   rocblas_int n, 
                   rocblas_int lda, 
                   rocblas_int incx,
                   rocblas_int ldc, 
                   bool verbose)
{
    rocblas_int size_a = lda * n;
    rocblas_int size_c = ldc * n;
    rocblas_int incx_pos = incx > 0 ? incx : -incx;
    rocblas_int size_x = incx_pos * (side == rocblas_side_right ? n : m);

    std::vector<T>ha(size_a);
    std::vector<T>hc(size_c);
    std::vector<T>hx(size_x);
    std::vector<T>hc_legacy(size_c);
    std::vector<T>hc_rocblas(size_c);

    initialize_matrix(ha, m, n, lda);
    initialize_matrix(hc, m, n, ldc);
    initialize_vector(hx, (side == rocblas_side_right ? n : m), incx_pos); 

    hc_legacy = hc;
    hc_rocblas = hc_legacy;

    if(verbose)
    {
        print_matrix("a", ha, lda, n, lda);

        print_matrix("c", hc, ldc, n, ldc);

        print_vector("x", hx, (side == rocblas_side_right ? n : m), incx_pos);
    }

    rocblas_status status;

    CHECK_ROCBLAS_ERROR( dgmm_reference( side,
              m, n, 
              ha.data(), lda,
              hx.data(), incx,
              hc_legacy.data(), ldc));

    CHECK_ROCBLAS_ERROR( rocblas_dgmm(side,
              m, n, 
              ha.data(), lda,
              hx.data(), incx,
              hc_rocblas.data(), ldc));

    if(verbose)
    {
        print_matrix("c: reference", hc_legacy, ldc, n, ldc);
        print_matrix("c: rocblas_dgmm", hc_rocblas, ldc, n, ldc);
    }

    T norm_err_rocblas_dgmm = 0.0;
    T norm_ref = 0.0;
    T tolerance = 100;
    T eps = std::numeric_limits<T>::epsilon();
    for (int i1 = 0; i1 < m; i1++)
    {
        for (int i2 = 0; i2 < n; i2++)
        {
            T t = hc_rocblas[i1+i2*ldc] - hc_legacy[i1+i2*ldc];
            norm_err_rocblas_dgmm += t * t;
	    if(t != t)
	    {
	        std::cout << "i1, i2, t, norm_err_rocblas_dgmm = " << i1 << ", " << i2 << ", " 
		    << t << ", " << norm_err_rocblas_dgmm << std::endl;
	    }

            norm_ref += hc_legacy[i1+i2*ldc] * hc_legacy[i1+i2*ldc];
        }
    }
    norm_err_rocblas_dgmm = sqrt(norm_err_rocblas_dgmm);
    norm_ref = sqrt(norm_ref);

    if (norm_err_rocblas_dgmm < norm_ref * eps * tolerance)
    {
        std::cout << "PASS ";
    }
    else
    {
        std::cout << "FAIL, norm_ref * eps * tol = " << norm_ref * eps * tolerance << std::endl;
    }
    
    std::cout << "norm_err = " << norm_err_rocblas_dgmm << std::endl;
}

void sdgmm(rocblas_side side, 
        rocblas_int m, 
        rocblas_int n, 
        rocblas_int lda, 
        rocblas_int incx,
        rocblas_int ldc, 
        bool verbose)
{
    template_dgmm<float>( side, m, n, lda, incx, ldc, verbose);
}

void ddgmm(rocblas_side side, 
        rocblas_int m, 
        rocblas_int n, 
        rocblas_int lda, 
        rocblas_int incx,
        rocblas_int ldc, 
        bool verbose)
{
    template_dgmm<double>( side, m, n, lda, incx, ldc, verbose);
}


int main(int argc, char* argv[])
{
    rocblas_side side = rocblas_side_left;

    rocblas_int m = 4, n = 3;
    rocblas_int lda = 0, ldc = 0, incx = 1;

    bool verbose = false;
    char precision = 's';

    if(parse_arguments(argc, argv, side, m, n, lda, ldc, incx, verbose, precision))
    {
        show_usage(argv);
        return EXIT_FAILURE;
    }
    ldc = ldc < m ? m : ldc;
    lda = lda < m ? m : lda;

    if(precision == 's' || precision == 'S')
    {
        sdgmm( side, m, n, lda, incx, ldc, verbose);
    }
    else if(precision == 'd' || precision == 'D')
    {
        ddgmm( side, m, n, lda, incx, ldc, verbose);
    }
//  else if(precision == 'c' || precision == 'C')
//  {
//      cdgmm( side, m, n, lda, incx, ldc);
//  }
//  else if(precision == 'z' || precision == 'Z')
//  {
//      zdgmm( side, diag, m, n, incx, lda, ldc);
//  }

    std::cout << " m,n,lda,incx,ldc,side,precision = " << m << ", " << n << ", " << lda << ", " << incx << ", " << ldc << ", ";

    side == rocblas_side_left ? std::cout << "left" : std::cout << "right"; 
    std::cout << ", " << (char)precision << std::endl;
    return 0;
}
