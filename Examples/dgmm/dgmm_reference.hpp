      template<typename T> rocblas_status dgmm_reference (
              rocblas_side side, 
              rocblas_int m, 
              rocblas_int n, 
              T *a, 
              rocblas_int lda, 
              T *x, 
              rocblas_int incx, 
              T *c, 
              rocblas_int ldc)
{
//  C <- A x   or   C <- x A
    if (rocblas_side_right == side)
    {
        for (int j = 0; j < n; j++)
        {
            for (int i = 0; i < m; i++)
            {
                c[i + j * ldc] = a[i + j * lda] * x[j * incx];
            }
        }
    }
    else
    {
        for (int j = 0; j < n; j++)
        {
            for (int i = 0; i < m; i++)
            {
                c[i + j * ldc] = a[i + j * lda] * x[i * incx];
            }
        }
    }

    return rocblas_status_success;
}
