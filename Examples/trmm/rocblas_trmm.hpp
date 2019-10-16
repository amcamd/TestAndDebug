#include <algorithm>
#include "rocblas.h"
//#include "rocblas-types.h"

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

template<typename T>
rocblas_status (*rocblas_gemm)(rocblas_handle handle,
        rocblas_operation transA,
        rocblas_operation transB,
        rocblas_int m,
        rocblas_int n,
        rocblas_int k,
        const T* alpha,
        const T* a,
        rocblas_int lda,
        const T* b,
        rocblas_int ldb,
        const T* beta,
        T* c,
        rocblas_int ldc);

template<>
static constexpr auto rocblas_gemm<float> = rocblas_sgemm;

template<>
static constexpr auto rocblas_gemm<double> = rocblas_dgemm;


template<typename T>
rocblas_status (*rocblas_gemv)(rocblas_handle handle,
        rocblas_operation trans,
        rocblas_int m,
        rocblas_int n,
        const T* alpha,
        const T* a,
        rocblas_int lda,
        const T* x,
        rocblas_int incx,
        const T* beta,
        T* y,
        rocblas_int incy);

template<>
static constexpr auto rocblas_gemv<float> = rocblas_sgemv;

template<>
static constexpr auto rocblas_gemv<double> = rocblas_dgemv;


__global__ void copy_void_ptr_vector_kernel_cut_and_paste(rocblas_int n,
                                            rocblas_int elem_size,
                                            const void* x,
                                            rocblas_int incx,
                                            void*       y,
                                            rocblas_int incy)
{
    size_t tid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    if(tid < n)
    {
        memcpy( (char*)y + tid * incy * elem_size, (const char*)x + tid * incx * elem_size, elem_size);
    }
}



template<typename T> 
rocblas_status copy_reference_device_device(
        rocblas_int n, 
        T* source, rocblas_int source_inc, 
        T* dest, rocblas_int dest_inc)
{
    constexpr rocblas_int NB_X               = 256;

    int  blocks = (n - 1) / NB_X + 1; // parameters for device kernel
    dim3 grid(blocks);
    dim3 threads(NB_X);

    rocblas_int elem_size = sizeof(T);

    hipLaunchKernelGGL(copy_void_ptr_vector_kernel_cut_and_paste,
                                           grid,
                                           threads,
                                           0,
                                           0,
                                           n,
                                           elem_size,
                                           source,
                                           source_inc,
                                           dest,
                                           dest_inc);
    return rocblas_status_success;
}


template<typename T> 
rocblas_status trmm_gemm_based_rocblas( 
        rocblas_handle handle,
        rocblas_side side, 
        rocblas_fill uplo, 
        rocblas_operation trans, 
        rocblas_diagonal diag,
        rocblas_int m, rocblas_int n, T *alpha,
        T *a, rocblas_int lda,
        T *c, rocblas_int ldc)
{
//
// Level 3 Blas routine.
//
// -- Written on 8-February-1989.
//    Jack Dongarra, Argonne National Laboratory.
//    iain Duff, AERE Harwell.
//    Jeremy Du Croz, Numerical Algorithms Group Ltd.
//    Sven Hammarling, Numerical Algorithms Group Ltd.
//
// -- Rewritten in December-1993.
//    GEMM-Based Level 3 BLAS.
//    Per Ling, institute of information Processing,
//    University of Umea, Sweden.
//

//  rocblas_int rb = 64, cb = 64, rcb = 64;
    rocblas_int rb =  4, cb =  4, rcb =  4;
    rocblas_int offd = rocblas_diagonal_unit == diag ? 1 : 0;
    rocblas_int nrowa = rocblas_side_left == side ? m : n;
    rocblas_int isec, jsec, tsec;
    T gamma, delta;
//  T t1[rb*cb];
//  T t2[cb*cb];
    T zero = 0.0;
    T one = 1.0;
    rocblas_int ldt1 = rb, ldt2 = cb;
    rocblas_status status = rocblas_status_success;


    T *dt1, *dt2;
    rocblas_int size_t1 = rb*cb;
    rocblas_int size_t2 = cb*cb;
    CHECK_HIP_ERROR(hipMalloc(&dt1, size_t1 * sizeof(T)));
    CHECK_HIP_ERROR(hipMalloc(&dt2, size_t2 * sizeof(T)));

//
//    Test the input parameters.
//
    if (side == rocblas_side_left)
    {
        nrowa = m;
    }
    else
    {
        nrowa= n;
    }

    if (m < 0 || n < 0 || lda < nrowa || ldc < m)
    {
        return rocblas_status_invalid_size;
    }
//
//  Quick return if possible.
//
    if (m == 0 || n == 0) return rocblas_status_success;

//
//
//    And when alpha.eq.zero.
//
      if (*alpha == 0)
      {
          CHECK_ROCBLAS_ERROR(rocblas_gemm<T>(handle,
                  rocblas_operation_none, rocblas_operation_none, 
                  m, n, 0, &zero, 
                  c, lda > ldc ? lda : ldc, 
                  c, lda > ldc ? lda : ldc, &zero, 
                  c, ldc));
          return rocblas_status_success;
      }
//
//    Start the operations.
//
    if (side == rocblas_side_left)
    {
        if (uplo == rocblas_fill_upper)
        {
            if (trans == rocblas_operation_none)
            {
//
//              Form  C := alpha*A*C. Left, Upper, No transpose.
//
                delta = *alpha;
                bool cldc = dcld(ldc);
                for (int ii = 1; ii <= m; ii += cb)
                {
                    isec = cb < m - ii + 1 ? cb : m - ii + 1;
//
//                  T2 := A', the transpose of a upper unit or non-unit
//                  triangular diagonal block of A is copied to the
//                  lower triangular part of T2.
//
                    for (int i = ii + offd; i <= ii + isec -1 ; i++)
                    {
                        copy_reference_device_device(i-ii+1-offd, &a[ii-1 + (i-1)*lda], 1, &dt2[i-ii], cb);
                    }
                    for (int jj = 1; jj <= n; jj += rb)
                    {
                        jsec = rb < n - jj + 1 ? rb : n - jj + 1;
//
//                      T1 := C', the transpose of a rectangular block
//                      of C is copied to T1.
//
                        if (cldc)
                        {
                            for (int j = jj; j <= jj + jsec -1; j++)
                            {
                                copy_reference_device_device(isec, &c[ii-1 + (j-1)*ldc], 1, &dt1[j-jj], rb);
                            }
                        }
                        else
                        {
                            for (int i = ii; i <= ii + isec -1; i++)
                            {
                                copy_reference_device_device(jsec, &c[i-1+(jj-1)*ldc], ldc, &dt1[(i-ii)*ldt1], 1);
                            }
                        }
//
//                      T1 := gamma*T1*T2 + delta*T1, triangular matrix
//                      multiply where the value of delta depends on
//                      whether T2 stores a unit or non-unit triangular
//                      block. Gamma and tsec are used to compensate for
//                      a deficiency in DGEMV that appears if the second
//                      dimension (tsec) is zero.
//
                        for (int i = ii; i <= ii + isec -1; i++)
                        {
                           if (diag == rocblas_diagonal_non_unit)
                           {
                               T hdiag;
                               CHECK_HIP_ERROR( hipMemcpy(&hdiag, &dt2[i-ii +(i-ii)*ldt2], sizeof(T), hipMemcpyDeviceToHost));
                               delta = *alpha * hdiag;
                           }
                           gamma = *alpha;
                           tsec = ii+isec-1-i;
                           if (tsec == 0)
                           {
                               tsec = 1;
                               gamma = 0.0;
                           }
                           CHECK_ROCBLAS_ERROR(rocblas_gemv<T>(handle,
                                   rocblas_operation_none, 
                                   jsec, tsec, &gamma, 
                                   &dt1[(i-ii+1)*ldt1], rb, 
                                   &dt2[i - ii + 1 + (i - ii)*ldt2], 1, &delta, 
                                   &dt1[(i-ii)*ldt1], 1));
                        }
//
//                      C := T1', the transpose of T1 is copied back
//                      to C.
//
                        for (int j = jj; j <= jj + jsec -1; j++)
                        {
                           copy_reference_device_device(isec, &dt1[j-jj], rb, &c[ii-1 + (j-1)*ldc], 1);
                        }
                    }
//
//                  C := alpha*A*C + C, general matrix multiply
//                  involving a rectangular block of A.
//
                    if( ii+isec <= m)
                    {
                        CHECK_ROCBLAS_ERROR(rocblas_gemm<T>(handle,
                                rocblas_operation_none, rocblas_operation_none, 
                                isec, n, m-ii-isec+1, alpha, 
                                &a[ii-1+(ii+isec-1)*lda], lda, 
                                &c[ii+isec-1], ldc, &one, 
                                &c[ii-1], ldc));
                    }
                }
            }
            else
            {
//
//             Form  C := alpha*A'*C. Left, Upper, Transpose.
//
                 delta = *alpha;
                 bool cldc = dcld(ldc);
                 for (int ii = m - ((m - 1) % cb); ii >= 1; ii -= cb)
                 {
                    isec = cb < m - ii + 1 ? cb : m - ii + 1;
                    for (int jj = 1; jj <= n; jj += rb)
                    {
                       jsec = rb < n - jj + 1 ? rb : n - jj + 1;
//
//                      T1 := C', the transpose of a rectangular block
//                      of C is copied to T1.
//
                       if (cldc)
                       {
                          for (int j = jj; j <= jj + jsec -1; j++)
                          {
                             copy_reference_device_device(isec, &c[ii-1 + (j-1)*ldc], 1, &dt1[j-jj], rb);
                          }
                       }
                       else
                       {
                          for (int i = ii; i <= ii + isec -1; i++)
                          {
                             copy_reference_device_device(jsec, &c[i-1+(jj-1)*ldc], ldc, &dt1[(i-ii)*ldt1], 1);
                          }
                       }
//
//                      T1 := gamma*T1*A + delta*T1, triangular matrix
//                      multiply where the value of delta depends on
//                      whether A is a unit or non-unit triangular
//                      matrix. Gamma and tsec are used to compensate
//                      for a deficiency in DGEMV that appears if the
//                      second dimension (tsec) is zero.
//
                       for (int i = ii + isec - 1; i >= ii; i--)
                       {
                          if (diag == rocblas_diagonal_non_unit)
                          {
                               T hdiag;
                               CHECK_HIP_ERROR( hipMemcpy(&hdiag, &a[i-1 +(i-1)*lda], sizeof(T), hipMemcpyDeviceToHost));
                               delta = *alpha * hdiag;
                          }
                          gamma = *alpha;
                          tsec = i-ii;
                          if (0 == tsec)
                          {
                             tsec = 1;
                             gamma = zero;
                          }
                          CHECK_ROCBLAS_ERROR(rocblas_gemv<T>(handle,
                                  rocblas_operation_none,
                                  jsec, tsec, &gamma, 
                                  dt1, rb,
                                  &a[ii-1 + (i-1)*lda], 1, &delta,
                                  &dt1[(i-ii)*ldt1], 1));
                       }
//
//                      C := T1', the transpose of T1 is copied back
//                      to C.
//
                       for (int j = jj; j <= jj + jsec -1; j++)
                       {
                          copy_reference_device_device(isec, &dt1[j-jj], rb, &c[ii-1 + (j-1)*ldc], 1);
                       }
                    }
//
//                   C := alpha*A'*C + C, general matrix multiply
//                   involving the transpose of a rectangular block
//                   of A.
//
                    if (ii > 1)
                    {
                        CHECK_ROCBLAS_ERROR(rocblas_gemm<T>(handle,
                               rocblas_operation_transpose, rocblas_operation_none,
                               isec, n, ii-1, alpha,
                               &a[(ii-1) * lda], lda,
                               c, ldc, &one,
                               &c[ii-1], ldc));
                    }
                }
            }
        }
        else
        {
            if (trans == rocblas_operation_none)
            {
//
//             Form  C := alpha*A*C. Left, Lower, No transpose.
//
                 delta = *alpha;
                 bool cldc = dcld(ldc);
                 for (int ix = m; ix >= 1; ix -= cb)
                 {
                    rocblas_int ii = 1 > ix-cb+1 ? 1 : ix-cb+1;
                    isec = ix-ii+1;
//
//                   T2 := A', the transpose of a lower unit or non-unit
//                   triangular diagonal block of A is copied to the
//                   upper triangular part of T2.
//
                    for (int i = ii; i <= ii + isec -1 - offd; i++)
                    {
                        copy_reference_device_device(ii+isec-i-offd, &a[i+offd-1 + (i-1)*lda], 1, &dt2[i-ii + (i-ii+offd)*ldt2], cb );
                    }
                    for (int jj = 1; jj <= n; jj += rb)
                    {
                       jsec = rb < n - jj + 1 ? rb : n - jj + 1;
//
//                      T1 := C', the transpose of a rectangular block
//                      of C is copied to T1.
//
                       if (cldc)
                       {
                          for (int j = jj; j <= jj + jsec -1; j++)
                          {
                             copy_reference_device_device(isec, &c[ii-1 + (j-1)*ldc], 1, &dt1[j-jj], rb);
                          }
                       }
                       else
                       {
                          for (int i = ii; i <= ii + isec - 1; i++)
                          {
                             copy_reference_device_device(jsec, &c[i-1+(jj-1)*ldc], ldc, &dt1[(i-ii)*ldt1], 1);
                          }
                       }
//
//                      T1 := gamma*T1*T2 + delta*T1, triangular matrix
//                      multiply where the value of delta depends on
//                      whether T2 stores a unit or non-unit triangular
//                      block. Gamma and tsec are used to compensate for
//                      a deficiency in DGEMV that appears if the second
//                      dimension (tsec) is zero.
//
                       for (int i = ii + isec - 1; i >= ii; i--)
                       {
                          if (diag == rocblas_diagonal_non_unit)
                          {
                               T hdiag;
                               CHECK_HIP_ERROR( hipMemcpy(&hdiag, &dt2[i-ii +(i-ii)*ldt2], sizeof(T), hipMemcpyDeviceToHost));
                               delta = *alpha * hdiag;
                          }
                          gamma = *alpha;
                          tsec = i-ii;
                          if (tsec == 0)
                          {
                             tsec = 1;
                             gamma = zero;
                          }
                          CHECK_ROCBLAS_ERROR(rocblas_gemv<T>(handle,
                              rocblas_operation_none,
                                  jsec, tsec, &gamma, 
                                  dt1, rb,
                                  &dt2[(i-ii)*ldt2], 1, &delta,
                                  &dt1[(i-ii)*ldt1], 1));
                       }
//
//                      C := T1', the transpose of T1 is copied back
//                      to C.
//
                       for (int j = jj; j <= jj + jsec -1; j++)
                       {
                          copy_reference_device_device(isec, &dt1[j-jj], rb, &c[ii-1 + (j-1)*ldc], 1);
                       }
                    }
//
//                   C := alpha*A'*C + C, general matrix multiply
//                   involving a rectangular block of A.
//
                    if (ii > 1)
                    {
                        CHECK_ROCBLAS_ERROR(rocblas_gemm<T>(handle,
                            rocblas_operation_none, rocblas_operation_none,
                                isec, n, ii-1, alpha,
                                &a[ii-1], lda,
                                c, ldc, &one,
                                &c[ii-1], ldc));
                    }
                 }
            }
            else
            {
//
//              Form  C := alpha*A'*C. Left, Lower, Transpose.
//
                delta = *alpha;
                bool cldc = dcld(ldc);
                for (int ix = ((m-1) % cb) + 1; ix <= m; ix += cb)
                {
                    rocblas_int ii = 1 > ix-cb+1 ? 1 : ix-cb+1;
                    isec = ix-ii+1;
                    for (int jj = 1; jj <= n; jj += rb)
                    {
                        jsec = rb < n - jj + 1 ? rb : n - jj + 1;
//
//                      T1 := C', the transpose of a rectangular block
//                      of C is copied to T1.
//
                        if (cldc)
                        {
                           for (int j = jj; j <= jj + jsec -1; j++)
                           {
                              copy_reference_device_device(isec, &c[ii-1 + (j-1)*ldc], 1, &dt1[j-jj], rb);
                           }
                        }
                        else
                        {
                           for (int i = ii; i <= ii + isec -1; i++)
                           {
                              copy_reference_device_device(jsec, &c[i-1+(jj-1)*ldc], ldc, &dt1[(i-ii)*ldt1], 1);
                           }
                        }
//
//                      T1 := gamma*T1*A + delta*T1, triangular matrix
//                      multiply where the value of delta depends on
//                      whether A is a unit or non-unit triangular
//                      matrix. Gamma and tsec are used to compensate
//                      for a deficiency in DGEMV that appears if the
//                      second dimension (tsec) is zero.
//
                        for (int i = ii; i <= ii + isec -1; i++)
                        {
                           if (diag == rocblas_diagonal_non_unit)
                           {
                               T hdiag;
                               CHECK_HIP_ERROR( hipMemcpy(&hdiag, &a[i-1 +(i-1)*lda], sizeof(T), hipMemcpyDeviceToHost));
                               delta = *alpha * hdiag;
// TODO: gives error
//                            delta = *alpha * a[i-1 + (i-1)*lda];
                           }
                           gamma = *alpha;
                           tsec = ii+isec-1-i;
                           if (tsec == 0)
                           {
                              tsec = 1;
                              gamma = zero;
                           }
                          CHECK_ROCBLAS_ERROR(rocblas_gemv<T>(handle,
                               rocblas_operation_none,
                                   jsec, tsec, &gamma,
                                   &dt1[(i-ii+1)*ldt1], rb,
                                   &a[i + (i-1)*lda], 1, &delta,
                                   &dt1[(i-ii)*ldt1], 1));
                        }
//
//                      C := T1', the transpose of T1 is copied back
//                      to C.
//
                        for (int j = jj; j <= jj + jsec -1; j++)
                        {
                           copy_reference_device_device(isec, &dt1[j-jj], rb, &c[ii-1 + (j-1)*ldc], 1);
                        }
                    }
//
//                  C := alpha*A'*C + C, general matrix multiply
//                  involving the transpose of a rectangular block
//                  of A.
//
                    if( ii+isec <= m)
                    {
                        CHECK_ROCBLAS_ERROR(rocblas_gemm<T>(handle,
                            rocblas_operation_transpose, rocblas_operation_none,
                                isec, n, m-ii-isec+1, alpha,
                                &a[ii+isec-1 + (ii-1)*lda], lda,
                                &c[ii+isec-1], ldc, &one,
                                &c[ii-1], ldc));
                    }
                }
            }
        }
    }
    else
    {
        if (uplo == rocblas_fill_upper)
        {
            if (trans == rocblas_operation_none)
            {
//
//              Form  C := alpha*C*A. Right, Upper, No transpose.
//
                delta = *alpha;
                for (int jj = n - (n-1) % cb; jj >= 1; jj -= cb)
                {
                    jsec = cb < n - jj + 1 ? cb : n - jj + 1;
                    for (int ii = 1; ii <= m; ii += rb)
                    {
                        isec = rb < m-ii+1 ? rb : m-ii+1;
//
//                      T1 := C, a rectangular block of C is copied
//                      to T1.
//
                        for (int j = jj; j <= jj + jsec -1; j++)
                        {
                           copy_reference_device_device( isec, &c[ii-1 + (j-1)*ldc], 1, &dt1[(j-jj)*ldt1], 1 );
                        }
//
//                      C := gamma*T1*A + delta*C, triangular matrix
//                      multiply where the value of delta depends on
//                      whether A is a unit or non-unit triangular
//                      matrix. Gamma and tsec are used to compensate
//                      for a deficiency in DGEmV that appears if the
//                      second dimension (tsec) is zero.
//
                        for (int j = jj+jsec-1; j >= jj; j--)
                        {
                            if (diag == rocblas_diagonal_non_unit)
                            {
                               T hdiag;
                               CHECK_HIP_ERROR( hipMemcpy(&hdiag, &a[j-1 +(j-1)*lda], sizeof(T), hipMemcpyDeviceToHost));
                               delta = *alpha * hdiag;
//                              delta = *alpha*a[j-1 + (j-1)*lda];
                            }
                            gamma = *alpha;
                            tsec = j - jj;
                            if (tsec == 0)
                            {
                                tsec = 1;
                                gamma = zero;
                            }
                          CHECK_ROCBLAS_ERROR(rocblas_gemv<T>(handle,
                                rocblas_operation_none, 
                                   isec, tsec, &gamma, 
                                   dt1, rb, 
                                   &a[jj-1 + (j-1)*lda], 1, &delta, 
                                   &c[ii-1 + (j-1)*ldc], 1));
                        }
                    }
//
//                  C := alpha*C*A + C, general matrix multiply
//                  involving a rectangular block of A.
//
                    if (jj > 1)
                    {
                        CHECK_ROCBLAS_ERROR(rocblas_gemm<T>(handle,
                            rocblas_operation_none, rocblas_operation_none, 
                                m, jsec, jj-1, alpha, 
                                c, ldc, 
                                &a[(jj-1)*lda], lda, &one, 
                                &c[(jj-1)*ldc], ldc));
                    }
                }
            }
            else
            {
//
//              Form  C := alpha*C*A'. Right, Upper, Transpose.
//
                delta = *alpha;
                for (int jj = 1; jj <= n; jj += cb)
                {
                    jsec = cb < n-jj+1 ? cb : n-jj+1;
//
//                  T2 := A', the transpose of a upper unit or non-unit
//                  triangular diagonal block of A is copied to the
//                  lower triangular part of T2.
//
                    for (int j = jj + offd; j <= jj + jsec -1; j++)
                    {
//                      CALL DCOPY ( J-JJ+1-offd, &a[jj-1 + (j-1)*lda], 1, T2( J-JJ+1, 1 ), cb )
                            copy_reference_device_device(j-jj+1-offd, &a[jj-1 + (j-1)*lda], 1, &dt2[j-jj], cb);
                    }
                    for (int ii = 1; ii <= m; ii += rb)
                    {
                        isec = rb < m-ii+1 ? rb : m-ii+1;
//
//                      T1 := C, a rectangular block of C is copied
//                      to T1.
//
                        for (int j = jj; j <= jj + jsec - 1; j++)
                        {
                           copy_reference_device_device(isec, &c[ii-1 + (j-1)*ldc], 1, &dt1[(j-jj)*ldt1], 1);
                        }
//
//                      C := gamma*T1*T2 + delta*C, triangular matrix
//                      multiply where the value of delta depends on
//                      whether T2 is a unit or non-unit triangular
//                      matrix. Gamma and tsec are used to compensate
//                      for a deficiency in DGEmV that appears if the
//                      second dimension (tsec) is zero.
//
                        for (int j = jj; j <= jj + jsec - 1; j++)
                        {
                           if (diag == rocblas_diagonal_non_unit)
                           {
                               T hdiag;
                               CHECK_HIP_ERROR( hipMemcpy(&hdiag, &dt2[j-jj +(j-jj)*ldt2], sizeof(T), hipMemcpyDeviceToHost));
                               delta = *alpha * hdiag;
//                            delta = *alpha * dt2[j-jj + (j-jj)*ldt2];
                           }
                           gamma = *alpha;
                           tsec = jj+jsec-1-j;
                           if (tsec == 0)
                           {
                              tsec = 1;
                              gamma = zero;
                           }
                          CHECK_ROCBLAS_ERROR(rocblas_gemv<T>(handle,
                               rocblas_operation_none, 
                                   isec, tsec, &gamma,
                                   &dt1[(j-jj+1)*ldt1], rb,
                                   &dt2[j-jj+1 +(j-jj)*ldt2], 1, &delta, 
                                   &c[ii-1 + (j-1)*ldc], 1));
                        }
                    }
//
//                  C := alpha*C*A' + C, general matrix multiply
//                  involving the transpose of a rectangular block
//                  of A.
//
                    if( jj+jsec <= n)
                    {
                        CHECK_ROCBLAS_ERROR(rocblas_gemm<T>(handle,
                            rocblas_operation_none, rocblas_operation_transpose, 
                                m, jsec, n-jj-jsec+1, alpha, 
                                &c[(jj+jsec-1)*ldc], ldc,
                                &a[jj-1 + (jj+jsec-1)*lda], lda, &one,
                                &c[(jj-1)*ldc], ldc ));
                    }
                }
            }
        }
        else
        {
            if (trans == rocblas_operation_none)
            {
//
//              Form  C := alpha*C*A. Right, Lower, No transpose.
//
                delta = *alpha;
                for (int jx = ((n-1) % cb) + 1; jx <= n; jx += cb)
                {
                    rocblas_int jj = 1 > jx - cb + 1 ?  1 : jx - cb + 1;
                    jsec = jx - jj + 1;
                    for (int ii = 1; ii <= m; ii += rb)
                    {
                        isec = rb < m-ii+1 ? rb : m-ii+1;
//
//                      T1 := C, a rectangular block of C is copied
//                      to T1.
//
                        for (int j = jj; j <= jj + jsec -1; j++)
                        {
                           copy_reference_device_device (isec, &c[ii-1 + (j-1)*ldc], 1, &dt1[(j-jj)*ldt1], 1);
                        }
//
//                      C := gamma*T1*A + delta*C, triangular matrix
//                      multiply where the value of delta depends on
//                      whether A is a unit or non-unit triangular
//                      matrix. Gamma and tsec are used to compensate
//                      for a deficiency in DGEmV that appears if the
//                      second dimension (tsec) is zero.
//
                        for (int j = jj; j <= jj + jsec -1; j++)
                        {
                            if (diag == rocblas_diagonal_non_unit)
                            {
                               T hdiag;
                               CHECK_HIP_ERROR( hipMemcpy(&hdiag, &a[j-1 +(j-1)*lda], sizeof(T), hipMemcpyDeviceToHost));
                               delta = *alpha * hdiag;
// TODO: fails
//                              delta = *alpha*a[j-1 + (j-1)*lda];
                            }
                            gamma = *alpha;
                            tsec = jj+jsec-1-j;
                            if (tsec == 0)
                            {
                                tsec = 1;
                                gamma = zero;
                            }
                          CHECK_ROCBLAS_ERROR(rocblas_gemv<T>(handle,
                                rocblas_operation_none, isec, tsec, &gamma,
                                    &dt1[(j-jj+1)*ldt1], rb, 
                                    &a[j + (j-1)*lda], 1, &delta, 
                                    &c[ii-1 + (j-1)*ldc], 1));
                        }
                    }
//
//                   C := alpha*C*A + C, general matrix multiply
//                   involving a rectangular block of A.
//
                    if( jj+jsec <= n)
                    {
                        CHECK_ROCBLAS_ERROR(rocblas_gemm<T>(handle,
                            rocblas_operation_none, rocblas_operation_none, 
                                m, jsec, n-jj-jsec+1, alpha,
                                &c[(jj+jsec-1)*ldc], ldc,
                                &a[jj+jsec-1 + (jj-1)*lda], lda, &one,
                                &c[(jj-1)*ldc], ldc));
                    }
                }
            }
            else
            {
//
//              Form  C := alpha*C*A'. Right, Lower, Transpose.
//
                delta = *alpha;
                for (int jx = n; jx >= 1; jx -= cb)
                {
                    rocblas_int jj = 1 > jx - cb + 1 ?  1 : jx - cb + 1;
                    jsec = jx - jj + 1;
//
//                  T2 := A', the transpose of a lower unit or non-unit
//                  triangular diagonal block of A is copied to the
//                  upper triangular part of T2.
//
                    for (int j = jj; j <= jj + jsec -1 - offd; j++)
                    {
                        copy_reference_device_device (jj+jsec-j-offd, 
                                &a[j+offd-1 + (j-1)*lda], 1, 
                                &dt2[j-jj + (j-jj+offd)*ldt2], cb );
                    }
                    for (int ii = 1; ii <= m; ii += rb)
                    {
                        isec = rb < m-ii+1 ? rb : m-ii+1;
//
//                      T1 := C, a rectangular block of C is copied
//                      to T1.
//
                        for (int j = jj; j <= jj + jsec - 1; j++)
                        {
                           copy_reference_device_device (isec, 
                                   &c[ii-1 + (j-1)*ldc], 1,
                                   &dt1[(j-jj)*ldt1], 1 );
                        }
//
//                      C := gamma*T1*T2 + delta*C, triangular matrix
//                      multiply where the value of delta depends on
//                      whether T2 is a unit or non-unit triangular
//                      matrix. Gamma and tsec are used to compensate
//                      for a deficiency in DGEmV that appears if the
//                      second dimension (tsec) is zero.
//
                        for (int j = jj + jsec -1; j >= jj ; j--)
                        {
                            if (diag == rocblas_diagonal_non_unit)
                            {
                               T hdiag;
                               CHECK_HIP_ERROR( hipMemcpy(&hdiag, &dt2[j-jj +(j-jj)*ldt2], sizeof(T), hipMemcpyDeviceToHost));
                               delta = *alpha * hdiag;
//                            delta = *alpha*dt2[j-jj + (j-jj)*ldt2];
                            }
                            gamma = *alpha;
                            tsec = j - jj;
                            if (tsec == 0)
                            {
                                tsec = 1;
                                gamma = zero;
                            }
                          CHECK_ROCBLAS_ERROR(rocblas_gemv<T>(handle,
                                rocblas_operation_none, 
                                    isec, tsec, &gamma, 
                                    dt1, rb, 
                                    &dt2[(j-jj)*ldt2], 1, &delta, 
                                    &c[ii-1 + (j-1)*ldc], 1));
                        }
                    }
//
//                  C := alpha*C*A' + C, general matrix multiply involving the transpose of a rectangular block of A.
//
                    if (jj > 1)
                    {
                        CHECK_ROCBLAS_ERROR(rocblas_gemm<T>(handle,
                            rocblas_operation_none, rocblas_operation_transpose, 
                                m, jsec, jj-1, alpha, 
                                c, ldc, 
                                &a[jj-1], lda, &one, 
                                &c[(jj-1)*ldc], ldc ));
                    }
                }
            }
        }
    }
    return status;
}

template<typename T> 
rocblas_status rocblas_trmm( 
        rocblas_side side, 
        rocblas_fill uplo, 
        rocblas_operation trans, 
        rocblas_diagonal diag,
        rocblas_int m, rocblas_int n, T alpha,
        T *ha, rocblas_int lda,
        T *hb, rocblas_int ldb)
{
    std::cout << " |rocblas_trmm| ";

    rocblas_status status = rocblas_status_success;

    rocblas_int ka = (side == rocblas_side_left) ? m : n;

    rocblas_int size_a = lda * ka;
    rocblas_int size_b = ldb * n;

    T *da, *db;
    CHECK_HIP_ERROR(hipMalloc(&da, size_a * sizeof(T)));
    CHECK_HIP_ERROR(hipMalloc(&db, size_b * sizeof(T)));

    CHECK_HIP_ERROR( hipMemcpy(da, ha, sizeof(T) * size_a, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR( hipMemcpy(db, hb, sizeof(T) * size_b, hipMemcpyHostToDevice));

    rocblas_handle handle;
    CHECK_ROCBLAS_ERROR(rocblas_create_handle(&handle));

    status = trmm_gemm_based_rocblas(handle, side, uplo, trans, diag, m, n, &alpha, da, lda, db, ldb);

//  CHECK_HIP_ERROR( hipMemcpy(ha, da, sizeof(T) * size_a, hipMemcpyDeviceToHost));
    CHECK_HIP_ERROR( hipMemcpy(hb, db, sizeof(T) * size_b, hipMemcpyDeviceToHost));


    return status;
}







































