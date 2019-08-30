#include "rocblas-types.h"

template <typename T>
rocblas_status copy_reference(rocblas_int n, T* x, rocblas_int incx, T* y, rocblas_int incy)
{
    for (int i = 0; i < n; i++)
    {
        y[i * incy] = x[i * incx];
    }
    return rocblas_status_success;
}

template <typename T>
void mat_mat_mult(T alpha, T beta, int M, int N, int K, 
        T* a, int as1, int as2, 
        T* b, int bs1, int bs2, 
        T* c, int cs1, int cs2)
{
    std::cout << "---------- in mat_mat_mult ----------" << std::endl;
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
    std::cout << "---------- in gemm_reference ----------" << std::endl;
    rocblas_int as1, as2, bs1, bs2, cs1, cs2;
    cs1 = 1;             // stride in first index
    cs2 = ldc;           // stride in second index
    // set leading dimension and strides depending on transA and transB
    if( transA == rocblas_operation_none)
    {
        as1 = 1; as2 = lda;
    }
    else
    {
        as1 = lda; as2 = 1;
    }
    if( transB == rocblas_operation_none)
    {
        bs1 = 1; bs2 = ldb;
    }
    else
    {
        bs1 = ldb; bs2 = 1;
    }

    mat_mat_mult(alpha, beta, m, n, k, 
            a, as1, as2, 
            b, bs1, bs2, 
            c, cs1, cs2);

    return rocblas_status_success;
}


template<typename T>
rocblas_status symm_l3_reference(
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
    std::cout << "---------- in symm_l3_reference ----------" << std::endl;

    rocblas_int        rcb = 4, cb = 2;
    T* t1 = new T[ rcb * rcb ];

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
        rocblas_int ldab = lda > ldb ? lda : ldb;
        rocblas_int k = 0;
        return gemm_reference(rocblas_operation_none, rocblas_operation_none,
        m, n, k, alpha, 
        a, ldab, 
        b, ldab, beta, 
        c, ldc);
    }
    std::cout << "---------- in symm_l3_reference -2-2-2-2-2" << std::endl;

    if (side == rocblas_side_left)
    {
        //*
        //*        Form  C := alpha*A*B + beta*C.
        //*
        if (uplo == rocblas_fill_upper)
        {
            for(int ii=1; ii <= m; ii += rcb)
            {
                int isec = rcb < m - ii + 1 ? rcb : m - ii + 1;
                for (int i = ii; i <= ii + isec -1; i++)
                {
                    copy_reference(i-ii+1, &a[ii-1 + (i-1)*lda], 1, &t1[(i-ii)*rcb], 1);
                }

                for (int jj = ii; jj <= ii+isec-1; jj += cb)
                {
                    int jsec = cb < ii + isec - jj ? cb : ii + isec - jj;
                    for (int j = jj + 1; j <= ii + isec -1; j++)
                    {
                        copy_reference(jsec < j-jj ? jsec : j-jj, &a[jj-1 + (j-1)*lda], 1, &t1[j-ii +(jj-ii)*rcb], rcb);
                    }
                }

                gemm_reference(rocblas_operation_none, rocblas_operation_none,
                isec, n, isec, alpha,
                t1, rcb,
                &b[ii-1], ldb, beta,
                &c[ii-1], ldc);

                if(ii > 1)
                {
                    gemm_reference(rocblas_operation_transpose, rocblas_operation_none,
                    isec, n, ii-1, alpha,
                    &a[(ii-1)*lda], lda,
                    b, ldb, static_cast<T>(1.0),
                    &c[ii-1], ldc);
                }

                if(ii + isec <= m)
                {
                    gemm_reference(rocblas_operation_none, rocblas_operation_none,
                    isec, n, m-ii-isec+1, alpha,
                    &a[ii-1 + (ii+isec-1)*lda], lda,
                    &b[ii+isec-1], ldb, static_cast<T>(1.0),
                    &c[ii-1], ldc);
                }
            }
        }
    }
    return rocblas_status_success;
}
