
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
#include "symm_reference.hpp"

template<typename T>
rocblas_status symm_st_bat_ref(
     rocblas_side      side,
     rocblas_fill      uplo,
     rocblas_int       m,
     rocblas_int       n,
     T                 alpha,
     T                 *a,
     rocblas_int       lda, rocblas_stride stride_a,
     T                 *b,
     rocblas_int       ldb, rocblas_stride stride_b,
     T                 beta,
     T                 *c,
     rocblas_int       ldc, rocblas_stride stride_c,
     rocblas_int       batch_count)
{
    //*
    //*  Original from
    //*  -- Reference BLAS level3 routine (version 3.7.0) --
    //*  -- Reference BLAS is a software package provided by Univ. of Tennessee,    --
    //*  -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..--
    //*     December 2016
    //*
    //*  rewritten by converting to c code
    //*  =====================================================================
    //*
    T temp1, temp2;
    T ONE = 1.0, ZERO = 0.0;
    
    // Set nrowa as the number of rows of A.
    rocblas_int nrowa = side == rocblas_side_left ? m : n;

    // Test the input parameters.
    if (m < 0 || n < 0 || lda < nrowa || lda < 1 || ldb < m || ldb < 1 || ldc < m || ldc < 1) 
    {
        return rocblas_status_invalid_size;
    }

    // Quick return if possible.
    if ((m == 0) || (n == 0) || ((alpha == ZERO) && (beta == ONE)))
    {
        return rocblas_status_success;
    }

    for (int ib = 0; ib < batch_count; ib++)
    {

    // And when  alpha.eq.zero.
    if (alpha == ZERO) 
    {
        if (beta == ZERO) 
        {
            for(int j=0; j<n; j++)
            {
                for(int i = 0; i < m; i++)
                {
                    c[i + j*ldc + ib*stride_c] = ZERO;
                }
            }
        }
        else
        {
            for(int j=0; j<n; j++)
            {
                for(int i = 0; i < m; i++)
                {
                    c[i + j*ldc + ib*stride_c] = beta*c[i + j*ldc + ib*stride_c];
                }
            }
        }
        return rocblas_status_success;
    }
    //*
    //*     Start the operations.
    //*
    if (side == rocblas_side_left)
    {
        //*
        //*        Form  C := alpha*A*B + beta*C.
        //*
        if (uplo == rocblas_fill_upper)
        {
            for(int j=0; j<n; j++)
            {
                for(int i = 0; i < m; i++)
                {
                    temp1 = alpha*b[i + j*ldb + ib*stride_b];
                    temp2 = ZERO;
                    for (int k = 0; k < i; k++)
                    {
                        c[k + j*ldc + ib*stride_c] += temp1*a[k + i*lda + ib*stride_a];
                        temp2 += b[k + j*ldb + ib*stride_b]*a[k + i*lda + ib*stride_a];
                    }
                    if (beta == ZERO) 
                    {
                        c[i + j*ldc + ib*stride_c] = temp1*a[i + i*lda + ib*stride_a] + alpha*temp2;
                    }
                    else
                    {
                        c[i + j*ldc + ib*stride_c] = beta*c[i + j*ldc + ib*stride_c] + temp1*a[i + i*lda + ib*stride_a] + alpha*temp2;
                    }
                }
            }
        }
        else
        {
            for(int j=0; j<n; j++)
            {
                for (int i = m-1; i>=0; i--)
                {
                    temp1 = alpha*b[i + j*ldb + ib*stride_b];
                    temp2 = ZERO;
                    for (int k = i+1; k < m; k++)
                    {
                        c[k + j*ldc + ib*stride_c] += temp1*a[k + i*lda + ib*stride_a];
                        temp2 += b[k + j*ldb + ib*stride_b]*a[k + i*lda + ib*stride_a];
                    }
                    if (beta == ZERO) 
                    {
                        c[i + j*ldc + ib*stride_c] = temp1*a[i + i*lda + ib*stride_a] + alpha*temp2;
                    }
                    else
                    {
                        c[i + j*ldc + ib*stride_c] = beta*c[i + j*ldc + ib*stride_c] + temp1*a[i + i*lda + ib*stride_a] + alpha*temp2;
                    }
                }
            }
        }
    }
    else
    {
        //*
        //*        Form  C := alpha*B*A + beta*C.
        //*

        for(int j=0; j<n; j++)
        {
            temp1 = alpha*a[j + j*lda + ib*stride_a];
            if (beta == ZERO) 
            {
                for(int i = 0; i < m; i++)
                {
                    c[i + j*ldc + ib*stride_c] = temp1*b[i + j*ldb + ib*stride_b];
                }
            }
            else
            {
                for(int i = 0; i < m; i++)
                {
                    c[i + j*ldc + ib*stride_c] = beta*c[i + j*ldc + ib*stride_c] + temp1*b[i + j*ldb + ib*stride_b];
                }
            }
            for (int k=0; k < j; k++)
            {
                if (uplo == rocblas_fill_upper) 
                {
                    temp1 = alpha*a[k + j*lda + ib*stride_a];
                }
                else
                {
                    temp1 = alpha*a[j + k*lda + ib*stride_a];
                }
                for(int i = 0; i < m; i++)
                {
                    c[i + j*ldc + ib*stride_c] = c[i + j*ldc + ib*stride_c] + temp1*b[i + k*ldb + ib*stride_b];
                }
            }
            for(int k = j+1; k < n; k++)
            {
                if (uplo == rocblas_fill_upper)
                {
                    temp1 = alpha*a[j + k*lda + ib*stride_a];
                }
                else
                {
                    temp1 = alpha*a[k + j*lda + ib*stride_a];
                }
                for(int i = 0; i < m; i++)
                {
                    c[i + j*ldc + ib*stride_c] = c[i + j*ldc + ib*stride_c] + temp1*b[i + k*ldb + ib*stride_b];
                }
            }
        }
    }
    }

    return rocblas_status_success;
}


template<typename T>
rocblas_status symm_reference(
     rocblas_side      side,
     rocblas_fill      uplo,
     rocblas_int       m,
     rocblas_int       n,
     T                 alpha,
     T                 *a,
     rocblas_int       lda,
     T                 *b,
     rocblas_int       ldb,
     T                 beta,
     T                 *c,
     rocblas_int       ldc)
{
    //*
    //*  Original from
    //*  -- Reference BLAS level3 routine (version 3.7.0) --
    //*  -- Reference BLAS is a software package provided by Univ. of Tennessee,    --
    //*  -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..--
    //*     December 2016
    //*
    //*  rewritten by converting to c code
    //*  =====================================================================
    //*
    T temp1, temp2;
    T ONE = 1.0, ZERO = 0.0;
    
    // Set nrowa as the number of rows of A.
    rocblas_int nrowa = side == rocblas_side_left ? m : n;

    // Test the input parameters.
    if (m < 0 || n < 0 || lda < nrowa || lda < 1 || ldb < m || ldb < 1 || ldc < m || ldc < 1) 
    {
        return rocblas_status_invalid_size;
    }

    // Quick return if possible.
    if ((m == 0) || (n == 0) || ((alpha == ZERO) && (beta == ONE)))
    {
        return rocblas_status_success;
    }

    // And when  alpha.eq.zero.
    if (alpha == ZERO) 
    {
        if (beta == ZERO) 
        {
            for(int j=0; j<n; j++)
            {
                for(int i = 0; i < m; i++)
                {
                    c[i + j*ldc] = ZERO;
                }
            }
        }
        else
        {
            for(int j=0; j<n; j++)
            {
                for(int i = 0; i < m; i++)
                {
                    c[i + j*ldc] = beta*c[i + j*ldc];
                }
            }
        }
        return rocblas_status_success;
    }
    //*
    //*     Start the operations.
    //*
    if (side == rocblas_side_left)
    {
        //*
        //*        Form  C := alpha*A*B + beta*C.
        //*
        if (uplo == rocblas_fill_upper)
        {
            for(int j=0; j<n; j++)
            {
                for(int i = 0; i < m; i++)
                {
                    temp1 = alpha*b[i + j*ldb];
                    temp2 = ZERO;
                    for (int k = 0; k < i; k++)
                    {
                        c[k + j*ldc] += temp1*a[k + i*lda];
                        temp2 += b[k + j*ldb]*a[k + i*lda];
                    }
                    if (beta == ZERO) 
                    {
                        c[i + j*ldc] = temp1*a[i + i*lda] + alpha*temp2;
                    }
                    else
                    {
                        c[i + j*ldc] = beta*c[i + j*ldc] + temp1*a[i + i*lda] + alpha*temp2;
                    }
                }
            }
        }
        else
        {
            for(int j=0; j<n; j++)
            {
                for (int i = m-1; i>=0; i--)
                {
                    temp1 = alpha*b[i + j*ldb];
                    temp2 = ZERO;
                    for (int k = i+1; k < m; k++)
                    {
                        c[k + j*ldc] += temp1*a[k + i*lda];
                        temp2 += b[k + j*ldb]*a[k + i*lda];
                    }
                    if (beta == ZERO) 
                    {
                        c[i + j*ldc] = temp1*a[i + i*lda] + alpha*temp2;
                    }
                    else
                    {
                        c[i + j*ldc] = beta*c[i + j*ldc] + temp1*a[i + i*lda] + alpha*temp2;
                    }
                }
            }
        }
    }
    else
    {
        //*
        //*        Form  C := alpha*B*A + beta*C.
        //*

        for(int j=0; j<n; j++)
        {
            temp1 = alpha*a[j + j*lda];
            if (beta == ZERO) 
            {
                for(int i = 0; i < m; i++)
                {
                    c[i + j*ldc] = temp1*b[i + j*ldb];
                }
            }
            else
            {
                for(int i = 0; i < m; i++)
                {
                    c[i + j*ldc] = beta*c[i + j*ldc] + temp1*b[i + j*ldb];
                }
            }
            for (int k=0; k < j; k++)
            {
                if (uplo == rocblas_fill_upper) 
                {
                    temp1 = alpha*a[k + j*lda];
                }
                else
                {
                    temp1 = alpha*a[j + k*lda];
                }
                for(int i = 0; i < m; i++)
                {
                    c[i + j*ldc] = c[i + j*ldc] + temp1*b[i + k*ldb];
                }
            }
            for(int k = j+1; k < n; k++)
            {
                if (uplo == rocblas_fill_upper)
                {
                    temp1 = alpha*a[j + k*lda];
                }
                else
                {
                    temp1 = alpha*a[k + j*lda];
                }
                for(int i = 0; i < m; i++)
                {
                    c[i + j*ldc] = c[i + j*ldc] + temp1*b[i + k*ldb];
                }
            }
        }
    }

    return rocblas_status_success;
}

#ifdef INSTANTIATE_SYMM_REFERENCE
#error INSTANTIATE_SYMM_REFERENCE already defined
#endif

#define INSTANTIATE_SYMM_REFERENCE(T_)                                  \
template rocblas_status symm_reference<T_>( rocblas_side      side,     \
                                            rocblas_fill      uplo,     \
                                            rocblas_int       m,        \
                                            rocblas_int       n,        \
                                            T_                alpha,    \
                                            T_                *a,       \
                                            rocblas_int       lda,      \
                                            T_                *b,       \
                                            rocblas_int       ldb,      \
                                            T_                beta,     \
                                            T_                *c,       \
                                            rocblas_int       ldc);

INSTANTIATE_SYMM_REFERENCE(float)
INSTANTIATE_SYMM_REFERENCE(double)

#undef INSTANTIATE_SYMM_REFERENCE


#ifdef INSTANTIATE_SYMM_ST_BAT_REF
#error INSTANTIATE_SYMM_ST_BAT_REF already defined
#endif

#define INSTANTIATE_SYMM_ST_BAT_REF(T_)                                  \
template rocblas_status symm_st_bat_ref<T_>                              \
(                                                                        \
     rocblas_side      side,                                             \
     rocblas_fill      uplo,                                             \
     rocblas_int       m,                                                \
     rocblas_int       n,                                                \
     T_                alpha,                                            \
     T_                *a,                                               \
     rocblas_int       lda, rocblas_stride stride_a,                     \
     T_                *b,                                               \
     rocblas_int       ldb, rocblas_stride stride_b,                     \
     T_                beta,                                             \
     T_                *c,                                               \
     rocblas_int       ldc, rocblas_stride stride_c,                     \
     rocblas_int       batch_count);

INSTANTIATE_SYMM_ST_BAT_REF(float)
INSTANTIATE_SYMM_ST_BAT_REF(double)

#undef INSTANTIATE_SYMM_ST_BAT_REF
