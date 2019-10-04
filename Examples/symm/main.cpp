
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
#include "symm_reference.hpp"
#include "symm_l3_reference.hpp"

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
                a[i+j*lda] = dis(gen);
//              a[i+j*lda] = 2.0;
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
                   rocblas_int ldc,
                   bool verbose)
{

    rocblas_int ka = (side == rocblas_side_left) ? m : n;

    rocblas_int size_a = lda * ka;
    rocblas_int size_b = ldb * n;
    rocblas_int size_c = ldc * n;

    std::vector<T>ha(size_a);
    std::vector<T>hb(size_b);
    std::vector<T>hc_legacy(size_c);
    std::vector<T>hc_gemm_based(size_c);

    initialize_symmetric_matrix(ha, m, n, lda, side, uplo);
    initialize_matrix(hb, m, n, ldb);
    initialize_matrix(hc_legacy, m, n, ldc);

    hc_gemm_based = hc_legacy;
    
    T *da, *db, *dc;
    CHECK_HIP_ERROR(hipMalloc(&da, size_a * sizeof(T)));
    CHECK_HIP_ERROR(hipMalloc(&db, size_b * sizeof(T)));
    CHECK_HIP_ERROR(hipMalloc(&dc, size_c * sizeof(T)));

    CHECK_HIP_ERROR( hipMemcpy(da, ha.data(), sizeof(T) * size_a, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR( hipMemcpy(db, hb.data(), sizeof(T) * size_b, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR( hipMemcpy(dc, hc_legacy.data(), sizeof(T) * size_c, hipMemcpyHostToDevice));

    rocblas_handle handle;
    CHECK_ROCBLAS_ERROR(rocblas_create_handle(&handle));

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
        print_matrix("c_legacy", hc_legacy, ldc, n, ldc);
    }

    rocblas_status status;

    status = symm_reference( side, uplo, m, n, alpha,
        ha.data(), lda,
        hb.data(), ldb, beta,
        hc_legacy.data(), ldc);
 
    status = symm_l3_reference( side, uplo, m, n, alpha,
        ha.data(), lda,
        hb.data(), ldb, beta,
        hc_gemm_based.data(), ldc);

    // calculate error
    T error = 0.0;
    T magnitude = 0.0;
    T tolerance = 1;
    T epsilon = std::numeric_limits<T>::epsilon();
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < m; j++)
        {
            magnitude += hc_legacy[j + i * ldc] > T(0) ? hc_legacy[j + i * ldc] : - hc_legacy[j + i * ldc];
            T err = hc_legacy[j + i * ldc] - hc_gemm_based[j + i * ldc];
            error += err * err;
        }
    }
    if (error < epsilon * tolerance * magnitude)
    {
        std::cout << "----- pass ----- ";
        std::cout << "error, magnitude " << error << ", " << magnitude << std::endl;
    }
    else
    {
        std::cout << "----- fail ----- FAIL ----- error ------ ERROR -----";
        std::cout << "error, magnitude, epsilon * tolerance * magnitude = " << error << ", " << magnitude << ", " << epsilon * tolerance * magnitude << std::endl;
    }
  
    if(verbose)
    {
        print_matrix("output c_legacy", hc_legacy, ldc, n, ldc);
        print_matrix("output c_gemm_based", hc_gemm_based, ldc, n, ldc);
    }
}

void ssymm(rocblas_side side, 
        rocblas_fill uplo, 
        rocblas_int m, 
        rocblas_int n, 
        float alpha, 
        rocblas_int lda, 
        rocblas_int ldb, 
        float beta, 
        rocblas_int ldc,
        bool verbose)
{
    template_symm( side, uplo, m, n, alpha, lda, ldb, beta, ldc, verbose);
}

void dsymm(rocblas_side side, 
        rocblas_fill uplo, 
        rocblas_int m, 
        rocblas_int n, 
        float alpha_in, 
        rocblas_int lda, 
        rocblas_int ldb, 
        float beta_in, 
        rocblas_int ldc,
        bool verbose)
{
    double alpha = static_cast<double>(alpha_in);
    double beta = static_cast<double>(beta_in);
    template_symm( side, uplo, m, n, alpha, lda, ldb, beta, ldc, verbose);
}


int main(int argc, char* argv[])
{
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
        ssymm( side, uplo, m, n, alpha, lda, ldb, beta, ldc, verbose);
    }
    else if(precision == 'd' || precision == 'D')
    {
        std::cout << "double" << std::endl;
        dsymm( side, uplo, m, n, alpha, lda, ldb, beta, ldc, verbose);
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
