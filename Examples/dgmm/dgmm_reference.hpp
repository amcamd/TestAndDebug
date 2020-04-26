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
    // in case of negative incx shift pointer to end of data for negative indexing
    ptrdiff_t offset_x = (incx < 0) ? - ptrdiff_t(incx) * (n - 1) : 0;
    std::cout << "offset_x = " << offset_x << std::endl;

//  C <- A x   or   C <- x A
    if (rocblas_side_right == side)
    {
        for (int j = 0; j < n; j++)
        {
            for (int i = 0; i < m; i++)
            {
                c[i + j * ldc] = a[i + j * lda] * x[offset_x + j * incx];
            }
        }
    }
    else
    {
        for (int j = 0; j < n; j++)
        {
            for (int i = 0; i < m; i++)
            {
                c[i + j * ldc] = a[i + j * lda] * x[offset_x + i * incx];
            }
        }
    }

    return rocblas_status_success;
}
