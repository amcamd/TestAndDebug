
#include <hip/hip_runtime.h>
#include <omp.h>
#include <iostream>
#include <iomanip>
#include <math.h>

#include "rocblas.h"
#include "symm_reference.hpp"

template <typename T>
void print_matrix(
            const char* name, T* A, rocblas_int m, rocblas_int n, rocblas_int lda)
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

template<typename T>
void set_mat_zero(rocblas_int n1, rocblas_int n2, rocblas_int s1, rocblas_int s2, T *a)
{
    for (int i1 = 0; i1 < n1; i1++)
    {
        for(int i2 = 0; i2 < n2; i2++)
        {
            a[i1*s1+i2*s2] = T(0);
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
    rocblas_int a_s1 = transA == rocblas_operation_none ? 1 : lda;
    rocblas_int a_s2 = transA == rocblas_operation_none ? lda : 1;

    rocblas_int b_s1 = transB == rocblas_operation_none ? 1 : ldb;
    rocblas_int b_s2 = transB == rocblas_operation_none ? ldb : 1;

    rocblas_int c_s1 = 1, c_s2 = ldc;

    for(int i1 = 0; i1 < m; i1++)
    {
        for(int i2 = 0; i2 < n; i2++)
        {
            T t = 0.0;
            for(int i3 = 0; i3 < k; i3++)
            {
                t +=  a[i1*a_s1 + i3*a_s2] * b[i3*b_s1 + i2*b_s2];
            }
            c[i1*c_s1 + i2*c_s2] = beta * c[i1*c_s1 + i2*c_s2] + alpha * t ;
        }
    }

    return rocblas_status_success;
}

template <typename T>
rocblas_status gemm_st_bat_ref(rocblas_operation transA, rocblas_operation transB,
        rocblas_int m, rocblas_int n, rocblas_int k, T alpha,
        T* a, rocblas_int lda, rocblas_stride stride_a,
        T* b, rocblas_int ldb, rocblas_stride stride_b, T beta,
        T* c, rocblas_int ldc, rocblas_stride stride_c, rocblas_int batch_count)
{
    rocblas_int a_s1 = transA == rocblas_operation_none ? 1 : lda;
    rocblas_int a_s2 = transA == rocblas_operation_none ? lda : 1;

    rocblas_int b_s1 = transB == rocblas_operation_none ? 1 : ldb;
    rocblas_int b_s2 = transB == rocblas_operation_none ? ldb : 1;

    rocblas_int c_s1 = 1, c_s2 = ldc;

    for(int i4 = 0; i4 < batch_count; i4++)
    {
        for(int i1 = 0; i1 < m; i1++)
        {
            for(int i2 = 0; i2 < n; i2++)
            {
                T t = 0.0;
                for(int i3 = 0; i3 < k; i3++)
                {
                    t +=  a[i1*a_s1 + i3*a_s2 + i4*stride_a] * b[i3*b_s1 + i2*b_s2 + i4*stride_b];
                }
                c[i1*c_s1 + i2*c_s2 + i4*stride_c] = beta * c[i1*c_s1 + i2*c_s2 + i4*stride_c] + alpha * t ;
            }
        }
    }

    return rocblas_status_success;
}

// should work for batched and strided_batched, only tested for batch_count==1
template<typename T>
rocblas_status symm_block_recursive( bool verbose, rocblas_side side, rocblas_fill uplo,
      rocblas_int m, rocblas_int n, T alpha, 
      T *a, rocblas_int lda,
      T *b, rocblas_int ldb, T beta,
      T *c, rocblas_int ldc)
{
    rocblas_int ka = rocblas_side_left == side ? m : n; // dimension of triangle matrix a

    rocblas_int nb_diag = 2;   // size of diag blocks of triangle matrix a

    rocblas_int n_nb = ka / nb_diag;   // number of diag blocks of matrix a of size nb_diag
    rocblas_int nb_rem = ka % nb_diag; // remainder diag block size when ka not multiple of nb_diag

    rocblas_int symm_m = rocblas_side_left == side ? nb_diag : m;  // diag block symm argument m
    rocblas_int symm_n = rocblas_side_left == side ?  n : nb_diag; // diag block symm argument n

    rocblas_int diag_a_stride = 1 + lda; // stride for diag blocks in a
    rocblas_int diag_b_stride = rocblas_side_left == side ? 1 : ldb; // stride of b panels
    rocblas_int diag_c_stride = rocblas_side_left == side ? 1 : ldc; // stride of c panels

    rocblas_int i_diag;  // index of diag block

    bool zero_c = false;  // debug parameter
    // calls to symm for diagonal blocks of size nb_diag
    for (int i_nb = 0; i_nb < n_nb; i_nb++)
    {
        i_diag = i_nb * nb_diag; // diag block at a[i_diag, i_diag], size is nb_diag

        if(zero_c)set_mat_zero(m, n, 1, ldc, c);   // for debugging
        symm_reference(side, uplo, symm_m, symm_n, alpha,
                &(a[i_diag * diag_a_stride]), lda,
                &(b[i_diag * diag_b_stride]), ldb, beta,
                &(c[i_diag * diag_c_stride]), ldc);
        if(verbose)print_matrix("c after symm 1", c, m, n, ldc);   // for debugging
    }

    // calls to symm for remainder diag block of size nb_rem, where nb_rem < nb_diag
    if(nb_rem != 0)
    {
        i_diag = n_nb * nb_diag; // diag block at a[i_diag, i_diag], size is nb_rem
        symm_m = rocblas_side_left == side ? nb_rem : m;
        symm_n = rocblas_side_left == side ? n : nb_rem;

        if(zero_c)set_mat_zero(m, n, 1, ldc, c); // for debugging
        symm_reference(side, uplo, symm_m, symm_n, alpha,
                &(a[i_diag * diag_a_stride]), lda,
                &(b[i_diag * diag_b_stride]), ldb, beta,
                &(c[i_diag * diag_c_stride]), ldc);
        if(verbose)print_matrix("c after rem symm ", c, m, n, ldc);
    }

    rocblas_int stride, stride_rem, i_start;
    rocblas_int nb_start = nb_diag; // starting size of sub-diag blocks in matrix a
    rocblas_int nb;  // size of sub-diagonal blocks of matrix a

    // calls to gemm for sub-diagonal square blocks in matrix a with size m = n = nb. 
    // Start with nb = nb_start. Each iteration of the outer loop nb doubles, and the 
    // number of gemm calls halves.
    for (nb = nb_start, i_start = nb_start; i_start < ka; i_start += nb, nb *= 2)
    {
        stride = nb * 2; // stride for both indices of a, and for one index of b and c
        n_nb = (ka - i_start) / stride; // number of calls to gemm
        stride_rem  = (ka - i_start) % stride; // remainder when stride not multiple of ka-istart
        if(stride_rem >= nb)
        {
            stride_rem = 0;
            n_nb += 1;
        }
    
        // sub-diagonal gemm blocks calls
        for(int i = 0; i < n_nb; i++)
        {
            rocblas_int i1 = i_start + (i * stride);
            rocblas_int i2 = i1 - nb;

            if(rocblas_side_right == side)
            {
                if(rocblas_fill_lower == uplo)
                {
                    // lower sub-diagonal (from stored part of a)
                    if(zero_c)set_mat_zero(m, n, 1, ldc, c);
                    gemm_reference(rocblas_operation_none, rocblas_operation_none, m, nb, nb, alpha,
                        &(b[i1*ldb]), ldb,
                        &(a[i1 + i2*lda]), lda, T(1.0),
                        &(c[i2*ldc]), ldc);
                    if(verbose)print_matrix("right, lower, c after gemm1", c, m, n, ldc);
    
                    // upper sub-diagonal (from transpose of stored part of a)
                    if(zero_c)set_mat_zero(m, n, 1, ldc, c);
                    gemm_reference(rocblas_operation_none, rocblas_operation_transpose, m, nb, nb, alpha,
                        &(b[i2*ldb]), ldb,
                        &(a[i1 + i2*lda]), lda, T(1.0),
                        &(c[i1*ldc]), ldc);
                    if(verbose)print_matrix("right, lower, c after gemm2", c, m, n, ldc);
                }
                else
                {
                    // upper sub-diagonal (from stored part of a)
                    if(zero_c)set_mat_zero(m, n, 1, ldc, c);
                    gemm_reference(rocblas_operation_none, rocblas_operation_none, m, nb, nb, alpha,
                        &(b[i2*ldb]), ldb,
                        &(a[i2 + i1*lda]), lda, T(1.0),
                        &(c[i1*ldc]), ldc);
                    if(verbose)print_matrix("right, upper, c after gemm1", c, m, n, ldc);

                    // lower sub-diagonal (from transpose of stored part of a)
                    if(zero_c)set_mat_zero(m, n, 1, ldc, c);
                    gemm_reference(rocblas_operation_none, rocblas_operation_transpose, m, nb, nb, alpha,
                        &(b[i1*ldb]), ldb,
                        &(a[i2 + i1*lda]), lda, T(1.0),
                        &(c[i2*ldc]), ldc);
                    if(verbose)print_matrix("right, upper, c after gemm2", c, m, n, ldc);
                }
            }
            else
            {
                if(rocblas_fill_lower == uplo)
                {
                    // lower sub-diagonal (from stored part of a)
                    if(zero_c)set_mat_zero(m, n, 1, ldc, c);
                    gemm_reference(rocblas_operation_none, rocblas_operation_none, nb, n, nb, alpha,
                        &(a[i1 + i2*lda]), lda,
                        &(b[i2]), ldb, T(1.0),
                        &(c[i1]), ldc);
                    if(verbose)print_matrix("left, lower, c after gemm1", c, m, n, ldc);
    
                    // upper sub-diagonal (from transpose of stored part of a)
                    if(zero_c)set_mat_zero(m, n, 1, ldc, c);
                    gemm_reference(rocblas_operation_transpose, rocblas_operation_none, nb, n, nb, alpha,
                        &(a[i1 + i2*lda]), lda,
                        &(b[i1]), ldb, T(1.0),
                        &(c[i2]), ldc);
                    if(verbose)print_matrix("left, lower, c after gemm2", c, m, n, ldc);
                }
                else
                {
                    // upper sub-diagonal (from stored part of a)
                    if(zero_c)set_mat_zero(m, n, 1, ldc, c);
                    gemm_reference(rocblas_operation_none, rocblas_operation_none, nb, n, nb, alpha,
                        &(a[i2 + i1*lda]), lda,
                        &(b[i1]), ldb, T(1.0),
                        &(c[i2]), ldc);
                    if(verbose)print_matrix("left, upper, c after gemm1", c, m, n, ldc);
    
                    // lower sub-diagonal (from transpose of stored part of a)
                    if(zero_c)set_mat_zero(m, n, 1, ldc, c);
                    gemm_reference(rocblas_operation_transpose, rocblas_operation_none, nb, n, nb, alpha,
                        &(a[i2 + i1*lda]), lda,
                        &(b[i2]), ldb, T(1.0),
                        &(c[i1]), ldc);
                    if(verbose)print_matrix("left, upper, c after gemm2", c, m, n, ldc);
                }
            }
        }
    
        // remainder gemm block of size nb_rem x nb where n_rem < nb
        if(stride_rem != 0)
        {
            rocblas_int i1 = i_start + n_nb * stride;
            rocblas_int i2 = i1 - nb;
            rocblas_int nb_rem = ka - i1;

            if(rocblas_side_right == side)
            {
                if(rocblas_fill_lower == uplo)
                {
                    // lower sub-diagonal (from stored part of a)
                    if(zero_c)set_mat_zero(m, n, 1, ldc, c);
                    gemm_reference(rocblas_operation_none, rocblas_operation_none, m, nb, nb_rem, alpha,
                        &(b[i1*ldb]), ldb,
                        &(a[i1 + i2*lda]), lda, T(1.0),
                        &(c[i2*ldc]), ldc);
                    if(verbose)print_matrix("right, lower, c after rem gemm1", c, m, n, ldc);
    
                    // upper sub-diagonal (from transpose of stored part of a)
                    if(zero_c)set_mat_zero(m, n, 1, ldc, c);
                    gemm_reference(rocblas_operation_none, rocblas_operation_transpose, m, nb_rem, nb, alpha,
                        &(b[i2*ldb]), ldb,
                        &(a[i1 + i2*lda]), lda, T(1.0),
                        &(c[i1*ldc]), ldc);
                    if(verbose)print_matrix("right, lower, c after rem gemm2", c, m, n, ldc);
                }
                else
                {
                    // upper sub-diagonal (from stored part of a)
                    if(zero_c)set_mat_zero(m, n, 1, ldc, c);
                    gemm_reference(rocblas_operation_none, rocblas_operation_none, m, nb_rem, nb, alpha,
                        &(b[i2*ldb]), ldb,
                        &(a[i2 + i1*lda]), lda, T(1.0),
                        &(c[i1*ldc]), ldc);
                    if(verbose)print_matrix("right, upper, c after rem gemm1", c, m, n, ldc);

                    // lower sub-diagonal (from transpose of stored part of a)
                    if(zero_c)set_mat_zero(m, n, 1, ldc, c);
                    gemm_reference(rocblas_operation_none, rocblas_operation_transpose, m, nb, nb_rem, alpha,
                        &(b[i1*ldb]), ldb,
                        &(a[i2 + i1*lda]), lda, T(1.0),
                        &(c[i2*ldc]), ldc);
                    if(verbose)print_matrix("right, upper, c after rem gemm2", c, m, n, ldc);
                }
            }
            else
            {
                if(rocblas_fill_lower == uplo)
                {
                    // lower sub-diagonal (from stored part of a)
                    if(zero_c)set_mat_zero(m, n, 1, ldc, c);
                    gemm_reference(rocblas_operation_none, rocblas_operation_none, nb_rem, n, nb, alpha,
                        &(a[i1 + i2*lda]), lda,
                        &(b[i2]), ldb, T(1.0),
                        &(c[i1]), ldc);
                    if(verbose)print_matrix("left, lower, c after rem gemm1", c, m, n, ldc);
    
                    // upper sub-diagonal (from transpose of stored part of a)
                    if(zero_c)set_mat_zero(m, n, 1, ldc, c);
                    gemm_reference(rocblas_operation_transpose, rocblas_operation_none, nb, n, nb_rem, alpha,
                        &(a[i1 + i2*lda]), lda,
                        &(b[i1]), ldb, T(1.0),
                        &(c[i2]), ldc);
                    if(verbose)print_matrix("left, lower, c after rem gemm2", c, m, n, ldc);
                }
                else
                {
                    // upper sub-diagonal (from stored part of a)
                    if(zero_c)set_mat_zero(m, n, 1, ldc, c);
                    gemm_reference(rocblas_operation_none, rocblas_operation_none, nb, n, nb_rem, alpha,
                        &(a[i2 + i1*lda]), lda,
                        &(b[i1]), ldb, T(1.0),
                        &(c[i2]), ldc);
                    if(verbose)print_matrix("left, upper, c after rem gemm1", c, m, n, ldc);
    
                    // lower sub-diagonal (from transpose of stored part of a)
                    if(zero_c)set_mat_zero(m, n, 1, ldc, c);
                    gemm_reference(rocblas_operation_transpose, rocblas_operation_none, nb_rem, n, nb, alpha,
                        &(a[i2 + i1*lda]), lda,
                        &(b[i2]), ldb, T(1.0),
                        &(c[i1]), ldc);
                    if(verbose)print_matrix("left, upper, c after rem gemm2", c, m, n, ldc);
                }
            }

        }
    }
    return rocblas_status_success;
}



template<typename T>
rocblas_status symm_strided_block_recursive( bool verbose, rocblas_side side, rocblas_fill uplo,
      rocblas_int m, rocblas_int n, T alpha, 
      T *a, rocblas_int lda,
      T *b, rocblas_int ldb, T beta,
      T *c, rocblas_int ldc)
{
    rocblas_int ka = rocblas_side_left == side ? m : n; // dimension of triangle matrix a

    rocblas_int nb_diag = 2;   // size of diag blocks of triangle matrix a

    rocblas_int n_nb = ka / nb_diag;   // number of diag blocks of matrix a of size nb_diag
    rocblas_int nb_rem = ka % nb_diag; // remainder diag block size when ka not multiple of nb_diag

    rocblas_int symm_m = rocblas_side_left == side ? nb_diag : m;  // diag block symm argument m
    rocblas_int symm_n = rocblas_side_left == side ?  n : nb_diag; // diag block symm argument n

    rocblas_int diag_a_stride = 1 + lda; // stride for diag blocks in a
    rocblas_int diag_b_stride = rocblas_side_left == side ? 1 : ldb; // stride of b panels
    rocblas_int diag_c_stride = rocblas_side_left == side ? 1 : ldc; // stride of c panels

    rocblas_int i_diag;  // index of diag block

    bool zero_c = false;  // debug parameter
    // calls to symm_strided_batched for diagonal blocks of size nb_diag
    if(zero_c)set_mat_zero(m, n, 1, ldc, c);   // for debugging
    symm_st_bat_ref(side, uplo, symm_m, symm_n, alpha,
                    a, lda, nb_diag * diag_a_stride,
                    b, ldb, nb_diag * diag_b_stride, beta,
                    c, ldc, nb_diag * diag_c_stride, n_nb);
    if(verbose)print_matrix("c after symm 1", c, m, n, ldc);   // for debugging

    // calls to symm for remainder diag block of size nb_rem, where nb_rem < nb_diag
    if(nb_rem != 0)
    {
        i_diag = n_nb * nb_diag; // diag block at a[i_diag, i_diag], size is nb_rem
        symm_m = rocblas_side_left == side ? nb_rem : m;
        symm_n = rocblas_side_left == side ? n : nb_rem;

        if(zero_c)set_mat_zero(m, n, 1, ldc, c); // for debugging
        symm_reference(side, uplo, symm_m, symm_n, alpha,
                &(a[i_diag * diag_a_stride]), lda,
                &(b[i_diag * diag_b_stride]), ldb, beta,
                &(c[i_diag * diag_c_stride]), ldc);
        if(verbose)print_matrix("c after rem symm ", c, m, n, ldc);
    }

    rocblas_int stride, stride_rem, i_start;
    rocblas_int nb_start = nb_diag; // starting size of sub-diag blocks in matrix a
    rocblas_int nb;  // size of sub-diagonal blocks of matrix a

    // calls to gemm for sub-diagonal square blocks in matrix a with size m = n = nb. 
    // Start with nb = nb_start. Each iteration of the outer loop nb doubles, and the 
    // number of gemm calls halves.
    for (nb = nb_start, i_start = nb_start; i_start < ka; i_start += nb, nb *= 2)
    {
        stride = nb * 2; // stride for both indices of a, and for one index of b and c
        n_nb = (ka - i_start) / stride; // number of calls to gemm
        stride_rem  = (ka - i_start) % stride; // remainder when stride not multiple of ka-istart
        if(stride_rem >= nb)
        {
            stride_rem = 0;
            n_nb += 1;
        }

        rocblas_int i_start1 = i_start;
        rocblas_int i_start2 = i_start - nb;
        rocblas_int i1 = i_start;
        rocblas_int i2 = i_start - nb;

        if(rocblas_side_right == side)
        {
            if(rocblas_fill_lower == uplo)
            {
                // lower sub-diagonal (from stored part of a)
                if(zero_c)set_mat_zero(m, n, 1, ldc, c);

                gemm_st_bat_ref(rocblas_operation_none, rocblas_operation_none, m, nb, nb, alpha,
                    &(b[i1*ldb]), ldb, stride*ldb,
                    &(a[i1 + i2*lda]), lda, (stride + stride*lda), T(1.0),
                    &(c[i2*ldc]), ldc, stride*ldc, n_nb);
                if(verbose)print_matrix("right, lower, c after gemm1", c, m, n, ldc);

                // upper sub-diagonal (from transpose of stored part of a)
                if(zero_c)set_mat_zero(m, n, 1, ldc, c);
                gemm_st_bat_ref(rocblas_operation_none, rocblas_operation_transpose, m, nb, nb, alpha,
                    &(b[i2*ldb]), ldb, stride*ldb,
                    &(a[i1 + i2*lda]), lda, stride*(1+lda), T(1.0),
                    &(c[i1*ldc]), ldc, stride*ldc, n_nb);
                if(verbose)print_matrix("right, lower, c after gemm2", c, m, n, ldc);
            }
            else
            {
                // upper sub-diagonal (from stored part of a)
                if(zero_c)set_mat_zero(m, n, 1, ldc, c);
                gemm_st_bat_ref(rocblas_operation_none, rocblas_operation_none, m, nb, nb, alpha,
                    &(b[i2*ldb]), ldb, stride*ldb,
                    &(a[i1-nb + i1*lda]), lda, stride*(1+lda), T(1.0),
                    &(c[i1*ldc]), ldc, stride*ldc, n_nb);
                if(verbose)print_matrix("right, upper, c after gemm1", c, m, n, ldc);

                // lower sub-diagonal (from transpose of stored part of a)
                if(zero_c)set_mat_zero(m, n, 1, ldc, c);
                gemm_st_bat_ref(rocblas_operation_none, rocblas_operation_transpose, m, nb, nb, alpha,
                    &(b[i1*ldb]), ldb, stride*ldb,
                    &(a[i1-nb + i1*lda]), lda, stride*(1+lda), T(1.0),
                    &(c[i2*ldc]), ldc, stride*ldc, n_nb);
                if(verbose)print_matrix("right, upper, c after gemm2", c, m, n, ldc);
            }
        }
        else
        {
            if(rocblas_fill_lower == uplo)
            {
                // lower sub-diagonal (from stored part of a)
                if(zero_c)set_mat_zero(m, n, 1, ldc, c);
                gemm_st_bat_ref(rocblas_operation_none, rocblas_operation_none, nb, n, nb, alpha,
                    &(a[i1 + i2*lda]), lda, stride*(1+lda),
                    &(b[i2]), ldb, stride, T(1.0),
                    &(c[i1]), ldc, stride, n_nb);
                if(verbose)print_matrix("left, lower, c after gemm1", c, m, n, ldc);

                // upper sub-diagonal (from transpose of stored part of a)
                if(zero_c)set_mat_zero(m, n, 1, ldc, c);
                gemm_st_bat_ref(rocblas_operation_transpose, rocblas_operation_none, nb, n, nb, alpha,
                    &(a[i1 + i2*lda]), lda, stride*(1+lda),
                    &(b[i1]), ldb, stride, T(1.0),
                    &(c[i2]), ldc, stride, n_nb);
                if(verbose)print_matrix("left, lower, c after gemm2", c, m, n, ldc);
            }
            else
            {
                // upper sub-diagonal (from stored part of a)
                if(zero_c)set_mat_zero(m, n, 1, ldc, c);
                gemm_st_bat_ref(rocblas_operation_none, rocblas_operation_none, nb, n, nb, alpha,
                    &(a[i2 + i1*lda]), lda, stride*(1+lda),
                    &(b[i1]), ldb, stride, T(1.0),
                    &(c[i2]), ldc, stride, n_nb);
                if(verbose)print_matrix("left, upper, c after gemm1", c, m, n, ldc);

                // lower sub-diagonal (from transpose of stored part of a)
                if(zero_c)set_mat_zero(m, n, 1, ldc, c);
                gemm_st_bat_ref(rocblas_operation_transpose, rocblas_operation_none, nb, n, nb, alpha,
                    &(a[i2 + i1*lda]), lda, stride*(1+lda),
                    &(b[i2]), ldb, stride, T(1.0),
                    &(c[i1]), ldc, stride, n_nb);
                if(verbose)print_matrix("left, upper, c after gemm2", c, m, n, ldc);
            }
        }
    
        // remainder gemm block of size nb_rem x nb where n_rem < nb
        if(stride_rem != 0)
        {
            rocblas_int i1 = i_start + n_nb * stride;
            rocblas_int i2 = i1 - nb;
            rocblas_int nb_rem = ka - i1;

            if(rocblas_side_right == side)
            {
                if(rocblas_fill_lower == uplo)
                {
                    // lower sub-diagonal (from stored part of a)
                    if(zero_c)set_mat_zero(m, n, 1, ldc, c);
                    gemm_reference(rocblas_operation_none, rocblas_operation_none, m, nb, nb_rem, alpha,
                        &(b[i1*ldb]), ldb,
                        &(a[i1 + i2*lda]), lda, T(1.0),
                        &(c[i2*ldc]), ldc);
                    if(verbose)print_matrix("right, lower, c after rem gemm1", c, m, n, ldc);
    
                    // upper sub-diagonal (from transpose of stored part of a)
                    if(zero_c)set_mat_zero(m, n, 1, ldc, c);
                    gemm_reference(rocblas_operation_none, rocblas_operation_transpose, m, nb_rem, nb, alpha,
                        &(b[i2*ldb]), ldb,
                        &(a[i1 + i2*lda]), lda, T(1.0),
                        &(c[i1*ldc]), ldc);
                    if(verbose)print_matrix("right, lower, c after rem gemm2", c, m, n, ldc);
                }
                else
                {
                    // upper sub-diagonal (from stored part of a)
                    if(zero_c)set_mat_zero(m, n, 1, ldc, c);
                    gemm_reference(rocblas_operation_none, rocblas_operation_none, m, nb_rem, nb, alpha,
                        &(b[i2*ldb]), ldb,
                        &(a[i2 + i1*lda]), lda, T(1.0),
                        &(c[i1*ldc]), ldc);
                    if(verbose)print_matrix("right, upper, c after rem gemm1", c, m, n, ldc);

                    // lower sub-diagonal (from transpose of stored part of a)
                    if(zero_c)set_mat_zero(m, n, 1, ldc, c);
                    gemm_reference(rocblas_operation_none, rocblas_operation_transpose, m, nb, nb_rem, alpha,
                        &(b[i1*ldb]), ldb,
                        &(a[i2 + i1*lda]), lda, T(1.0),
                        &(c[i2*ldc]), ldc);
                    if(verbose)print_matrix("right, upper, c after rem gemm2", c, m, n, ldc);
                }
            }
            else
            {
                if(rocblas_fill_lower == uplo)
                {
                    // lower sub-diagonal (from stored part of a)
                    if(zero_c)set_mat_zero(m, n, 1, ldc, c);
                    gemm_reference(rocblas_operation_none, rocblas_operation_none, nb_rem, n, nb, alpha,
                        &(a[i1 + i2*lda]), lda,
                        &(b[i2]), ldb, T(1.0),
                        &(c[i1]), ldc);
                    if(verbose)print_matrix("left, lower, c after rem gemm1", c, m, n, ldc);
    
                    // upper sub-diagonal (from transpose of stored part of a)
                    if(zero_c)set_mat_zero(m, n, 1, ldc, c);
                    gemm_reference(rocblas_operation_transpose, rocblas_operation_none, nb, n, nb_rem, alpha,
                        &(a[i1 + i2*lda]), lda,
                        &(b[i1]), ldb, T(1.0),
                        &(c[i2]), ldc);
                    if(verbose)print_matrix("left, lower, c after rem gemm2", c, m, n, ldc);
                }
                else
                {
                    // upper sub-diagonal (from stored part of a)
                    if(zero_c)set_mat_zero(m, n, 1, ldc, c);
                    gemm_reference(rocblas_operation_none, rocblas_operation_none, nb, n, nb_rem, alpha,
                        &(a[i2 + i1*lda]), lda,
                        &(b[i1]), ldb, T(1.0),
                        &(c[i2]), ldc);
                    if(verbose)print_matrix("left, upper, c after rem gemm1", c, m, n, ldc);
    
                    // lower sub-diagonal (from transpose of stored part of a)
                    if(zero_c)set_mat_zero(m, n, 1, ldc, c);
                    gemm_reference(rocblas_operation_transpose, rocblas_operation_none, nb_rem, n, nb, alpha,
                        &(a[i2 + i1*lda]), lda,
                        &(b[i2]), ldb, T(1.0),
                        &(c[i1]), ldc);
                    if(verbose)print_matrix("left, upper, c after rem gemm2", c, m, n, ldc);
                }
            }

        }
    }
    return rocblas_status_success;
}

#ifdef INSTANTIATE_SYMM_BLOCK_RECURSIVE
#error INSTANTIATE_SYMM_BLOCK_RECURSIVE already defined
#endif

#define INSTANTIATE_SYMM_BLOCK_RECURSIVE(T_)                                                                \
template rocblas_status symm_block_recursive<T_>( bool verbose, rocblas_side side, rocblas_fill uplo,       \
                                                  rocblas_int m, rocblas_int n, T_ alpha,                   \
                                                  T_ *a, rocblas_int lda,                                   \
                                                  T_ *b, rocblas_int ldb, T_ beta,                          \
                                                  T_* c, rocblas_int ldc);

INSTANTIATE_SYMM_BLOCK_RECURSIVE(float)
INSTANTIATE_SYMM_BLOCK_RECURSIVE(double)

#undef INSTANTIATE_SYMM_BLOCK_RECURSIVE

#ifdef INSTANTIATE_SYMM_STRIDED_BLOCK_RECURSIVE
#error INSTANTIATE_SYMM_STRIDED_BLOCK_RECURSIVE already defined
#endif

#define INSTANTIATE_SYMM_STRIDED_BLOCK_RECURSIVE(T_)                                                                \
template rocblas_status symm_strided_block_recursive<T_>( bool verbose, rocblas_side side, rocblas_fill uplo,       \
                                                  rocblas_int m, rocblas_int n, T_ alpha,                   \
                                                  T_ *a, rocblas_int lda,                                   \
                                                  T_ *b, rocblas_int ldb, T_ beta,                          \
                                                  T_* c, rocblas_int ldc);

INSTANTIATE_SYMM_STRIDED_BLOCK_RECURSIVE(float)
INSTANTIATE_SYMM_STRIDED_BLOCK_RECURSIVE(double)

#undef INSTANTIATE_SYMM_STRIDED_BLOCK_RECURSIVE
