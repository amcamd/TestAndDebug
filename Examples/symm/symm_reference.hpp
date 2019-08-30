
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
            for(int k = j; k < n; k++)
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
