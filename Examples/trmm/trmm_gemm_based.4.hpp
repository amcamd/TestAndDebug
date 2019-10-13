#include <algorithm>
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
rocblas_status gemv_reference(rocblas_operation trans, rocblas_int m, rocblas_int n, T alpha,
        T* a, rocblas_int lda,
        T* x, rocblas_int incx, T beta,
        T* y, rocblas_int incy)
{
   rocblas_int s1, s2, n1, n2;
   if( trans == rocblas_operation_none)
   {   
       s1 = 1; s2 = lda;
       n1 = m;
       n2 = n;
   }
   else
   {   
       s1 = lda; s2 = 1;
       n1 = n;
       n2 = m;
   }
   for (int i1 = 0; i1 < n1; i1++)
   {
       T temp = 0.0;
       for (int i2 = 0; i2 < n2; i2++)
       {
           temp += a[i1 * s1 + i2 * s2] * x[i2 * incx];
       }
       y[i1 * incy] = alpha * temp + beta * y[i1 * incy];
   }
   return rocblas_status_success;
}

template <typename T>
rocblas_status gemm_reference(rocblas_operation transA, rocblas_operation transB, 
        rocblas_int m, rocblas_int n, rocblas_int k, T alpha, 
        T* a, rocblas_int lda, 
        T* b, rocblas_int ldb, T beta,
        T* c, rocblas_int ldc)
{
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
rocblas_status trmm_gemm_based_reference ( 
        rocblas_side side, 
        rocblas_fill uplo, 
        rocblas_operation trans, 
        rocblas_diagonal diag,
        rocblas_int m, rocblas_int n, T alpha,
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
    T t1[rb*cb];
    T t2[cb*cb];
    T t3[rcb*rcb];
    T zero = 0.0;
    T one = 1.0;
    rocblas_int ldt1 = rb, ldt2 = cb, ldt3 = rcb;
    rocblas_status status = rocblas_status_success;
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
      if (alpha == 0)
      {
         gemm_reference(rocblas_operation_none, rocblas_operation_none, m, n, 0, zero, c, lda > ldc ? lda : ldc, c, lda > ldc ? lda : ldc, zero, c, ldc);
         return status;
      }

//
//    Start the operations.
//
///   if (side == rocblas_side_left) {
///      if (uplo == rocblas_fill_upper) }
            if (trans == rocblas_operation_none)
            {
//
//             Form  C := alpha*A*C. Left, Upper, No transpose.
//
                  delta = alpha;
                  bool cldc = dcld(ldc);
                  for (int ii = 1; ii <= m; ii += cb)
                  {
                     isec = cb < m - ii + 1 ? cb : m - ii + 1;
//
//                   T2 := A', the transpose of a upper unit or non-unit
//                   triangular diagonal block of A is copied to the
//                   lower triangular part of T2.
//
                     for (int i = ii + offd; i <= ii + isec -1 ; i++)
                     {
                        copy_reference(i-ii+1-offd, &a[ii-1 + (i-1)*lda], 1, &t2[i-ii], cb);
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
                              copy_reference(isec, &c[ii-1 + (j-1)*ldc], 1, &t1[j-jj], rb);
                           }
                        }
                        else
                        {
                           for (int i = ii; i <= ii + isec -1; i++)
                           {
                              copy_reference(jsec, &c[i-1+(jj-1)*ldc], ldc, &t1[(i-ii)*ldt1], 1);
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
                              delta = alpha * t2[i-ii +(i-ii)*ldt2];
                           }
                           gamma = alpha;
                           tsec = ii+isec-1-i;
                           if (tsec == 0)
                           {
                              tsec = 1;
                              gamma = 0.0;
                           }
                           gemv_reference(rocblas_operation_none, jsec, tsec, gamma, &t1[(i-ii+1)*ldt1], rb, &t2[i - ii + 1 + (i - ii)*ldt2], 1, delta, &t1[(i-ii)*ldt1], 1);
                        }
//
//                      C := T1', the transpose of T1 is copied back
//                      to C.
//
                        for (int j = jj; j <= jj + jsec -1; j++)
                        {
                           copy_reference(isec, &t1[j-jj], rb, &c[ii-1 + (j-1)*ldc], 1);
                        }
                     }
//
//                   C := alpha*A*C + C, general matrix multiply
//                   involving a rectangular block of A.
//
                     if( ii+isec <= m)
                     {
                        gemm_reference(rocblas_operation_none, rocblas_operation_none, 
                                isec, n, m-ii-isec+1, alpha, 
                                &a[ii-1+(ii+isec-1)*lda], lda, 
                                &c[ii+isec-1], ldc, one, 
                                &c[ii-1], ldc);
                     }
                  }
           } else {
//
//             Form  C := alpha*A'*C. Left, Upper, Transpose.
//
                 std::cout << "left, upper, transpose" << std::endl;
                 delta = alpha;
                 bool cldc = dcld(ldc);
                 for (int ii = m - ((m - 1) % cb); ii >= 1; ii -= cb )
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
                             copy_reference(isec, &c[ii-1 + (j-1)*ldc], 1, &t1[j-jj], rb);
                          }
                       }
                       else
                       {
                          for (int i = ii; i <= ii + isec -1; i++ )
                          {
                             copy_reference(jsec, &c[i-1+(jj-1)*ldc], ldc, &t1[(i-ii)*ldt1], 1);
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
                             delta = alpha * a[i-1 + (i-1) * lda];
                          }
                          gamma = alpha;
                          tsec = i-ii;
                          if (0 == tsec)
                          {
                             tsec = 1;
                             gamma = zero;
                          }
                          gemv_reference(rocblas_operation_none,
                                  jsec, tsec, gamma, 
                                  t1, rb,
                                  &a[ii-1 + (i-1)*lda], 1, delta,
                                  &t1[(i-ii)*ldt1], 1);
                       }
//
//                      C := T1', the transpose of T1 is copied back
//                      to C.
//
                       for (int j = jj; j <= jj + jsec -1; j++)
                       {
                          copy_reference(isec, &t1[j-jj], rb, &c[ii-1 + (j-1)*ldc], 1);
                       }
                    }
//
//                   C := alpha*A'*C + C, general matrix multiply
//                   involving the transpose of a rectangular block
//                   of A.
//
                    if (ii > 1)
                    {
                       gemm_reference(rocblas_operation_transpose, rocblas_operation_none,
                               isec, n, ii-1, alpha,
                               &a[(ii-1) * lda], lda,
                               c, ldc, one,
                               &c[ii-1], ldc);
                    }
                 }
           }
      return rocblas_status_success;
}
///      } else {
///         if (trans == rocblas_operation_none) {
//
//             Form  C := alpha*A*C. Left, Lower, No transpose.
//
///               delta = alpha
///               cldc = DCLD( ldc )
///               for (int ix = m; ix >= 1; ix -= cb) {
///                  ii = MAX( 1, iX-cb+1 )
///                  isec = iX-ii+1
//
//                   T2 := A', the transpose of a lower unit or non-unit
//                   triangular diagonal block of A is copied to the
//                   upper triangular part of T2.
//
///                  for (int i = ii; i <= ii + isec -1 - offd; i++) {
///                     CALL DCOPY ( ii+isec-i-OFFD, A( i+OFFD, i ), 1, T2( i-ii+1, i-ii+1+OFFD ), cb )
///60                }
///                  for (int jj = 1; jj <= n; jj += rb) {
///                     jsec = rb < n - jj + 1 ? rb : n - jj + 1;
//
//                      T1 := C', the transpose of a rectangular block
//                      of C is copied to T1.
//
///                     if (cldc) {
///                        for (int j = jj; j <= jj + jsec -1; j++) {
///                           copy_reference(isec, &c[ii-1 + (j-1)*ldc], 1, &t1[j-jj], rb);
///70                      }
///                     } else {
///                        for (int i = ii; i <= ii + isec - 1; i++) {
///                           copy_reference(jsec, &c[i-1+(jj-1)*ldc], ldc, &t1[(i-ii)*ldt1], 1);
///80                      }
///                     }
//
//                      T1 := gamma*T1*T2 + delta*T1, triangular matrix
//                      multiply where the value of delta depends on
//                      whether T2 stores a unit or non-unit triangular
//                      block. Gamma and tsec are used to compensate for
//                      a deficiency in DGEMV that appears if the second
//                      dimension (tsec) is zero.
//
///                     for (int i = ii + isec - 1; i >= ii; i -= 1) {
///                        if (diag == rocblas_diagonal_non_unit) {
///                           delta = alpha * t2[i-ii +(i-ii)*ldt2];
///                        }
///                        gamma = alpha
///                        tsec = i-ii
///                        iF( tsec == 0   {
///                           tsec = 1
///                           gamma = ZERO
///                        }
///                        CALL DGEMV ( 'N', JSEC, tsec, gamma, T1( 1, 1 ), RB, T2( 1, i-ii+1 ), 1, delta, T1( 1, i-ii+1 ), 1 )
///90                   }
//
//                      C := T1', the transpose of T1 is copied back
//                      to C.
//
///                     for (int j = jj; j <= jj + jsec -1; j++) {
///                        copy_reference(isec, &t1[j-jj], rb, &c[ii-1 + (j-1)*ldc], 1);
///00                   }
///10                }
//
//                   C := alpha*A'*C + C, general matrix multiply
//                   involving a rectangular block of A.
//
///                  iF( ii.GT.1   {
///                     CALL DGEMM ( 'N', 'N', isec, N, ii-1, alpha, A( ii, 1 ), lda, C( 1, 1 ), ldc, ONE, C( ii, 1 ), ldc )
///                  }
///20             }
///         } else {
//
//             Form  C := alpha*A'*C. Left, Lower, Transpose.
//
///               delta = alpha
///               cldc = DCLD( ldc )
///               for (int ix = ((m-1) % cb) + 1; ix <= m; ix += cb) {
///                  ii = MAX( 1, iX-cb+1 )
///                  isec = iX-ii+1
///                  for (int jj = 1; jj <= n; jj += rb) {
///                     jsec = rb < n - jj + 1 ? rb : n - jj + 1;
//
//                      T1 := C', the transpose of a rectangular block
//                      of C is copied to T1.
//
///                     if (cldc) {
///                        for (int j = jj; j <= jj + jsec -1; j++) {
///                           copy_reference(isec, &c[ii-1 + (j-1)*ldc], 1, &t1[j-jj], rb);
///70                      }
///                     } else {
///                        for (int i = ii; i <= ii + isec -1; i++) {
///                           copy_reference(jsec, &c[i-1+(jj-1)*ldc], ldc, &t1[(i-ii)*ldt1], 1);
///80                      }
///                     }
//
//                      T1 := gamma*T1*A + delta*T1, triangular matrix
//                      multiply where the value of delta depends on
//                      whether A is a unit or non-unit triangular
//                      matrix. Gamma and tsec are used to compensate
//                      for a deficiency in DGEMV that appears if the
//                      second dimension (tsec) is zero.
//
///                     for (int i = ii; i <= ii + isec -1; i++) {
///                        if (diag == rocblas_diagonal_non_unit) {
///                           delta = alpha*A( i, i )
///                        }
///                        gamma = alpha
///                        tsec = ii+isec-1-i
///                        iF( tsec == 0   {
///                           tsec = 1
///                           gamma = ZERO
///                        }
///                        CALL DGEMV ( 'N', JSEC, tsec, gamma, T1( 1, i-ii+2 ), RB, A( i+1, i ), 1, delta, T1( 1, i-ii+1 ), 1 )
///90                   }
//
//                      C := T1', the transpose of T1 is copied back
//                      to C.
//
///                     for (int j = jj; j <= jj + jsec -1; j++) {
///                        copy_reference(isec, &t1[j-jj], rb, &c[ii-1 + (j-1)*ldc], 1);
///00                   }
///10                }
//
//                   C := alpha*A'*C + C, general matrix multiply
//                   involving the transpose of a rectangular block
//                   of A.
//
///                  iF( ii+isec <= M   {
///                     CALL DGEMM ( 'T', 'N', isec, N, M-ii-isec+1, alpha, A( ii+isec, ii ), lda, C( ii+isec, 1 ), ldc, ONE, C( ii, 1 ), ldc )
///                  }
///20             }
///         }
///      }
///   } else {
///      if (uplo == rocblas_fill_upper) }
///         if (trans == rocblas_operation_none) {
//
//             Form  C := alpha*C*A. Right, Upper, No transpose.
//
///               delta = alpha
///               for (int jj = n - (n-1) % cb; jj >= 1; jj -= cb) {
///                  JSEC = MiN( cb, N-JJ+1 )
///                  for (int ii = 1; ii <= m; ii += rb) {
///                     isec = MiN( RB, M-ii+1 )
//
//                      T1 := C, a rectangular block of C is copied
//                      to T1.
//
///                     for (int j = jj; j <= jj + jsec -1; j++) {
///                        CALL DCOPY ( isec, C( ii, J ), 1, T1( 1, J-JJ+1 ), 1 )
///50                   }
//
//                      C := gamma*T1*A + delta*C, triangular matrix
//                      multiply where the value of delta depends on
//                      whether A is a unit or non-unit triangular
//                      matrix. Gamma and tsec are used to compensate
//                      for a deficiency in DGEMV that appears if the
//                      second dimension (tsec) is zero.
//
///                     for (int j = jj+jsec-1; j >= jj; j--) {
///                        if (diag == rocblas_diagonal_non_unit) {
///                           delta = alpha*A( J, J )
///                        }
///                        gamma = alpha
///                        tsec = J-JJ
///                        iF( tsec == 0   {
///                           tsec = 1
///                           gamma = ZERO
///                        }
///                        CALL DGEMV ( 'N', isec, tsec, gamma, T1( 1, 1 ), RB, A( JJ, J ), 1, delta, C( ii, J ), 1 )
///60                   }
///70                }
//
//                   C := alpha*C*A + C, general matrix multiply
//                   involving a rectangular block of A.
//
///                  iF( JJ.GT.1   {
///                     CALL DGEMM ( 'N', 'N', M, JSEC, JJ-1, alpha, C( 1, 1 ), ldc, A( 1, JJ ), lda, ONE, C( 1, JJ ), ldc )
///                  }
///80             }
///         } else {
//
//             Form  C := alpha*C*A'. Right, Upper, Transpose.
//
///               delta = alpha
///               for (int jj = 1; jj <= n; jj += cb) {
///                  JSEC = MiN( cb, N-JJ+1 )
//
//                   T2 := A', the transpose of a upper unit or non-unit
//                   triangular diagonal block of A is copied to the
//                   lower triangular part of T2.
//
///                  for (int j = jj + offd; j <= jj + jsec -1; j++) {
///                     CALL DCOPY ( J-JJ+1-OFFD, A( JJ, J ), 1, T2( J-JJ+1, 1 ), cb )
///10                }
///                  for (int ii = 1; ii <= m; ii += rb) {
///                     isec = MiN( RB, M-ii+1 )
//
//                      T1 := C, a rectangular block of C is copied
//                      to T1.
//
///                     for (int j = jj; j <= jj + jsec - 1; j++) {
///                        CALL DCOPY ( isec, C( ii, J ), 1,
///  $                                              T1( 1, J-JJ+1 ), 1 )
///20                   }
//
//                      C := gamma*T1*T2 + delta*C, triangular matrix
//                      multiply where the value of delta depends on
//                      whether T2 is a unit or non-unit triangular
//                      matrix. Gamma and tsec are used to compensate
//                      for a deficiency in DGEMV that appears if the
//                      second dimension (tsec) is zero.
//
///                     for (int j = jj; j <= jj + jsec - 1; j++) {
///                        if (diag == rocblas_diagonal_non_unit) {
///                           delta = alpha*T2( J-JJ+1, J-JJ+1 )
///                        }
///                        gamma = alpha
///                        tsec = JJ+JSEC-1-J
///                        iF( tsec == 0   {
///                           tsec = 1
///                           gamma = ZERO
///                        }
///                        CALL DGEMV ( 'N', isec, tsec, gamma,
///  $                        T1( 1, J-JJ+2 ), RB, T2( J-JJ+2, J-JJ+1 ),
///  $                                         1, delta, C( ii, J ), 1 )
///30                   }
///40                }
//
//                   C := alpha*C*A' + C, general matrix multiply
//                   involving the transpose of a rectangular block
//                   of A.
//
///                  iF( JJ+JSEC <= N   {
///                     CALL DGEMM ( 'N', 'T', M, JSEC, N-JJ-JSEC+1,
///  $                                      alpha, C( 1, JJ+JSEC ), ldc,
///  $                                       A( JJ, JJ+JSEC ), lda, ONE,
///  $                                                 C( 1, JJ ), ldc )
///                  }
///50             }
///         }
///      } else {
///         if (trans == rocblas_operation_none) {
//
//             Form  C := alpha*C*A. Right, Lower, No transpose.
//
///               delta = alpha
///               for (int jx = ((n-1) % cb) + 1; jx <= n; jx += cb) {
///                  JJ = MAX( 1, JX-cb+1 )
///                  JSEC = JX-JJ+1
///                  for (int ii = 1; ii <= m; ii += rb) {
///                     isec = MiN( RB, M-ii+1 )
//
//                      T1 := C, a rectangular block of C is copied
//                      to T1.
//
///                     for (int j = jj; j <= jj + jsec -1; j++) {
///                        CALL DCOPY ( isec, C( ii, J ), 1,
///  $                                              T1( 1, J-JJ+1 ), 1 )
///80                   }
//
//                      C := gamma*T1*A + delta*C, triangular matrix
//                      multiply where the value of delta depends on
//                      whether A is a unit or non-unit triangular
//                      matrix. Gamma and tsec are used to compensate
//                      for a deficiency in DGEMV that appears if the
//                      second dimension (tsec) is zero.
//
///                     for (int j = jj; j <= jj + jsec -1; j++) {
///                        if (diag == rocblas_diagonal_non_unit) {
///                           delta = alpha*A( J, J )
///                        }
///                        gamma = alpha
///                        tsec = JJ+JSEC-1-J
///                        iF( tsec == 0   {
///                           tsec = 1
///                           gamma = ZERO
///                        }
///                        CALL DGEMV ( 'N', isec, tsec, gamma,
///  $                              T1( 1, J-JJ+2 ), RB, A( J+1, J ), 1,
///  $                                            delta, C( ii, J ), 1 )
///90                   }
///00                }
//
//                   C := alpha*C*A + C, general matrix multiply
//                   involving a rectangular block of A.
//
///                  iF( JJ+JSEC <= N   {
///                     CALL DGEMM ( 'N', 'N', M, JSEC, N-JJ-JSEC+1,
///  $                                      alpha, C( 1, JJ+JSEC ), ldc,
///  $                                       A( JJ+JSEC, JJ ), lda, ONE,
///  $                                                 C( 1, JJ ), ldc )
///                  }
///10             }
///         } else {
//
//             Form  C := alpha*C*A'. Right, Lower, Transpose.
//
///               delta = alpha
///               for (int jx = n; jx >= 1; jx -= cb) {
///                  JJ = MAX( 1, JX-cb+1 )
///                  JSEC = JX-JJ+1
//
//                   T2 := A', the transpose of a lower unit or non-unit
//                   triangular diagonal block of A is copied to the
//                   upper triangular part of T2.
//
///                  for (int j = jj; j <= jj + jsec -1 - offd; j++) {
///                     CALL DCOPY ( JJ+JSEC-J-OFFD, A( J+OFFD, J ),
///  $                                1, T2( J-JJ+1, J-JJ+1+OFFD ), cb )
///40                }
///                  for (int ii = 1; ii <= m; ii += rb) {
///                     isec = MiN( RB, M-ii+1 )
//
//                      T1 := C, a rectangular block of C is copied
//                      to T1.
//
///                     for (int j = jj; j <= jj + jsec - 1; j++) {
///                        CALL DCOPY ( isec, C( ii, J ), 1,
///  $                                              T1( 1, J-JJ+1 ), 1 )
///50                   }
//
//                      C := gamma*T1*T2 + delta*C, triangular matrix
//                      multiply where the value of delta depends on
//                      whether T2 is a unit or non-unit triangular
//                      matrix. Gamma and tsec are used to compensate
//                      for a deficiency in DGEMV that appears if the
//                      second dimension (tsec) is zero.
//
///                     for (int j = jj + jsec -1; j >= jj ; j-- ) {
///                        if (diag == rocblas_diagonal_non_unit) {
///                           delta = alpha*T2( J-JJ+1, J-JJ+1 )
///                        }
///                        gamma = alpha
///                        tsec = J-JJ
///                        iF( tsec == 0   {
///                           tsec = 1
///                           gamma = ZERO
///                        }
///                        CALL DGEMV ( 'N', isec, tsec, gamma, T1( 1, 1 ), RB, T2( 1, J-JJ+1 ), 1, delta, C( ii, J ), 1 )
///60                   }
///70                }
//
//                   C := alpha*C*A' + C, general matrix multiply
//                   involving the transpose of a rectangular block
//                   of A.
//
///                  iF( JJ.GT.1   {
///                     CALL DGEMM ( 'N', 'T', M, JSEC, JJ-1, alpha, C( 1, 1 ), ldc, A( JJ, 1 ), lda, ONE, C( 1, JJ ), ldc )
///                  }
///80             }
///         }
///      }
///   }
//
///   return status;
//
//    End of DTRMM.
//
///   END
//
