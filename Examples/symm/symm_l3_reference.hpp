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

    if (side == rocblas_side_left)
    {
        if (uplo == rocblas_fill_upper)
        {
            // Form  C := alpha*A*B + beta*C. Left, Upper.
            //
            for(int ii=1; ii <= m; ii += rcb)
            {
                int isec = std::min(rcb, m - ii + 1);
                for (int i = ii; i <= ii + isec -1; i++)
                {
                    copy_reference(i-ii+1, &a[ii-1 + (i-1)*lda], 1, &t1[(i-ii)*rcb], 1);
                }

                for (int jj = ii; jj <= ii+isec-1; jj += cb)
                {
                    int jsec = std::min(cb, ii + isec - jj);
                    for (int j = jj + 1; j <= ii + isec -1; j++)
                    {
                        copy_reference(std::min(jsec, j-jj), &a[jj-1 + (j-1)*lda], 1, &t1[j-ii +(jj-ii)*rcb], rcb);
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
        else
        {
            // Form  C := alpha*A*B + beta*C. Left, Lower.
            //
            for (int ix = m; ix >= 1; ix -= rcb)
            {
                int ii = std::max(1, ix - rcb + 1);
                int isec = ix - ii + 1;

                for (int i = ii; i <= ii + isec - 1; i++)
                {
                    copy_reference(ii + isec - i, &a[i-1 + (i-1)*lda], 1, &t1[i-ii + (i-ii) * rcb], 1);
                }

                for (int jx = ii + isec - 1; jx >= ii; jx -= cb)
                {
                    int jj = std::max(ii, jx - cb + 1);
                    int jsec = jx -jj + 1;
                    for (int j = ii; j <= jj + jsec - 2; j++)
                    {
                        copy_reference(std::min(jsec, jj + jsec - 1 - j),
                                &a[std::max(jj - 1, j) + (j -1)*lda], 1,
                                &t1[(j - ii) + std::max(jj - ii, j - ii + 1) * rcb], rcb);
                    }
                }

                gemm_reference (rocblas_operation_none, rocblas_operation_none,
                isec, n, isec, alpha,
                t1, rcb,
                &b[ii-1], ldb, beta,
                &c[ii-1], ldc);

                if (ii+isec <= m)
                {
                    gemm_reference(rocblas_operation_transpose, rocblas_operation_none,
                            isec, n, m - ii - isec + 1, alpha,
                            &a[ii+isec-1 + (ii-1)*lda], lda, 
                            &b[ii+isec-1], ldb, static_cast<T>(1.0), 
                            &c[ii-1], ldc );

                }

                if (ii > 1)
                {
                    gemm_reference(rocblas_operation_none, rocblas_operation_none,
                            isec, n, ii-1, alpha,
                            &a[ii - 1], lda, 
                            b, ldb, static_cast<T>(1.0),
                            &c[ii - 1], ldc );
                }
            }
        }    
    }
    else
    {
        if (uplo == rocblas_fill_upper)
        {
            // Form  C := alpha*B*A + beta*C. Right, Upper.
            //

            for (int jj = 1; jj <= n; jj += rcb)
            {
               int jsec = std::min( rcb, n-jj+1 );
//
//             T1 := A, a upper triangular diagonal block of A is copied
//             to the upper triangular part of T1.
//
               for (int j = jj; j <= jj + jsec -1; j++)
               {
                   copy_reference (j - jj + 1, &a[jj - 1 + lda * (j - 1)], 1, &t1[rcb * (j - jj)], 1);
               }
//
//             T1 :=  A', a strictly upper triangular diagonal block of
//             A is copied to the strictly lower triangular part of T1.
//             Notice that T1 is referenced by row and that the maximum
//             length of a vector referenced by DCOPY is CB.
//
               for (int ii = jj; ii <= jj + jsec - 1; ii += cb)
               {
                  int isec = std::min( cb, jj+jsec-ii );
                  for (int i = ii + 1; i <= jj + jsec - 1; i++)
                  {
                      copy_reference( std::min(isec, i - ii), &a[ii - 1 + lda * (i - 1)], 1, &t1[i - jj + rcb * (ii - jj)], rcb);
                  }
               }
//
//             C := alpha*B*T1 + beta*C, a vertical block of C is updated
//             using the general matrix multiply, DGEMM. T1 corresponds
//             to a full diagonal block of the matrix A.
//
               gemm_reference(rocblas_operation_none, rocblas_operation_none,
                       m, jsec, jsec, alpha,
                       &b[ldb * (jj - 1)], ldb,
                       t1, rcb, beta,
                       &c[ldc * (jj - 1)], ldc);


//
//             C := alpha*B*A + C and C := alpha*B*A' + C, general
//             matrix multiply operations involving rectangular blocks
//             of A.
//
               if (jj > 1)
               {
                   gemm_reference(rocblas_operation_none, rocblas_operation_none,
                           m, jsec, jj-1, alpha,
                           b, ldb,
                           &a[lda * (jj - 1)], lda, static_cast<T>(1.0),
                           &c[ldc * (jj - 1)], ldc);
               }
               if (jj - jsec <= n)
               {
                   gemm_reference(rocblas_operation_none, rocblas_operation_transpose,
                           m, jsec, n - jj - jsec + 1, alpha,
                           &b[ldb * (jj + jsec -1)], ldb,
                           &a[jj - 1 + lda * (jj + jsec -1)], lda, static_cast<T>(1.0),
                           &c[ldc * (jj-1)], ldc);
               }
            }
        }
        else
        {
            // Form  C := alpha*B*A + beta*C. Right, Lower.
            //
            for (int jx = n; jx >= 1; jx -= rcb)
            {
               int jj = std::max( 1, jx-rcb+1 );
               int jsec = jx-jj+1;
//
//             T1 := A, a lower triangular diagonal block of A is copied
//             to the lower triangular part of T1.
//
               for (int j = jj; j <= jj + jsec -1; j++)
               {
                  copy_reference( jj + jsec - j, &a[j - 1 + lda * (j - 1)], 1, &t1[j - jj + rcb * (j - jj)], 1);
               }
//
//             T1 :=  A', a strictly lower triangular diagonal block of
//             A is copied to the strictly upper triangular part of T1.
//             Notice that T1 is referenced by row and that the maximum
//             length of a vector referenced by DCOPY is CB.
//
               for (int ix = jj + jsec -1; ix >= jj; ix -= cb)
               {
                   int ii = std::max(jj, ix - cb + 1);
                   int isec = ix - ii + 1;
                   for (int i = jj; i <= ii + isec -2; i++)
                   {
                       copy_reference (std::min( isec, ii+isec-1-i ), 
                         &a[ std::max( ii-1, i ) + lda * (i-1) ], 1,
                         &t1[ i-jj + rcb * std::max( ii-jj, i-jj+1 ) ], rcb );
                   }
               }
//
//             C := alpha*B*T1 + beta*C, a vertical block of C is
//             updated using the general matrix multiply, DGEMM. T1
//             corresponds to a full diagonal block of the matrix A.
//
               gemm_reference(rocblas_operation_none, rocblas_operation_none,
                   m, jsec, jsec, alpha,
                   &b[ldb * (jj - 1)] , ldb, 
                   t1, rcb, beta, 
                   &c[ldc * (jj - 1)], ldc);
//
//             C := alpha*B*A + C and C := alpha*B*A' + C, general
//             matrix multiply operations involving rectangular blocks
//             of A.
//
               if( jj + jsec <= n )
               {
                   gemm_reference(rocblas_operation_none, rocblas_operation_none,
                       m, jsec, n-jj-jsec+1, alpha,
                       &b[ ldb * (jj + jsec - 1) ], ldb, 
                       &a[ jj + jsec - 1 + lda * (jj - 1) ], lda, static_cast<T>(1.0),
                       &c[ ldc * (jj - 1) ], ldc);

               }     
               if( jj > 1 )
               {
                   gemm_reference(rocblas_operation_none, rocblas_operation_transpose,
                       m, jsec, jj - 1, alpha,
                       b, ldb, 
                       &a[jj - 1], lda, static_cast<T>(1.0),
                       &c[ldc * (jj - 1)], ldc);
               }
            }
        }
    }
    return rocblas_status_success;
}


/*
      SUBROUTINE DSYMM( SIDE, UPLO, M, N, ALPHA, A, LDA, B, LDB,
     $                   BETA, C, LDC )
*     .. Scalar Arguments ..
      CHARACTER*1        SIDE, UPLO
      INTEGER            M, N, LDA, LDB, LDC
      DOUBLE PRECISION   ALPHA, BETA
*     .. Array Arguments ..
      DOUBLE PRECISION   A( LDA, * ), B( LDB, * ), C( LDC, * )
*     ..
*
*  Purpose
*  =======
*
*  DSYMM  performs one of the matrix-matrix operations
*
*     C := alpha*A*B + beta*C,
*
*  or
*
*     C := alpha*B*A + beta*C,
*
*  where alpha and beta are scalars,  A is a symmetric matrix and  B and
*  C are  m by n matrices.
*
*  Parameters
*  ==========
*
*  SIDE   - CHARACTER*1.
*           On entry,  SIDE  specifies whether  the  symmetric matrix  A
*           appears on the  left or right  in the  operation as follows:
*
*              SIDE = 'L' or 'l'   C := alpha*A*B + beta*C,
*
*              SIDE = 'R' or 'r'   C := alpha*B*A + beta*C,
*
*           Unchanged on exit.
*
*  UPLO   - CHARACTER*1.
*           On  entry,   UPLO  specifies  whether  the  upper  or  lower
*           triangular  part  of  the  symmetric  matrix   A  is  to  be
*           referenced as follows:
*
*              UPLO = 'U' or 'u'   Only the upper triangular part of the
*                                  symmetric matrix is to be referenced.
*
*              UPLO = 'L' or 'l'   Only the lower triangular part of the
*                                  symmetric matrix is to be referenced.
*
*           Unchanged on exit.
*
*  M      - INTEGER.
*           On entry,  M  specifies the number of rows of the matrix  C.
*           M  must be at least zero.
*           Unchanged on exit.
*
*  N      - INTEGER.
*           On entry, N specifies the number of columns of the matrix C.
*           N  must be at least zero.
*           Unchanged on exit.
*
*  ALPHA  - DOUBLE PRECISION.
*           On entry, ALPHA specifies the scalar alpha.
*           Unchanged on exit.
*
*  A      - DOUBLE PRECISION array of DIMENSION ( LDA, ka ), where ka is
*           m  when  SIDE = 'L' or 'l'  and is  n otherwise.
*           Before entry  with  SIDE = 'L' or 'l',  the  m by m  part of
*           the array  A  must contain the  symmetric matrix,  such that
*           when  UPLO = 'U' or 'u', the leading m by m upper triangular
*           part of the array  A  must contain the upper triangular part
*           of the  symmetric matrix and the  strictly  lower triangular
*           part of  A  is not referenced,  and when  UPLO = 'L' or 'l',
*           the leading  m by m  lower triangular part  of the  array  A
*           must  contain  the  lower triangular part  of the  symmetric
*           matrix and the  strictly upper triangular part of  A  is not
*           referenced.
*           Before entry  with  SIDE = 'R' or 'r',  the  n by n  part of
*           the array  A  must contain the  symmetric matrix,  such that
*           when  UPLO = 'U' or 'u', the leading n by n upper triangular
*           part of the array  A  must contain the upper triangular part
*           of the  symmetric matrix and the  strictly  lower triangular
*           part of  A  is not referenced,  and when  UPLO = 'L' or 'l',
*           the leading  n by n  lower triangular part  of the  array  A
*           must  contain  the  lower triangular part  of the  symmetric
*           matrix and the  strictly upper triangular part of  A  is not
*           referenced.
*           Unchanged on exit.
*
*  LDA    - INTEGER.
*           On entry, LDA specifies the first dimension of A as declared
*           in the calling (sub) program.  When  SIDE = 'L' or 'l'  then
*           LDA must be at least  max( 1, m ), otherwise  LDA must be at
*           least  max( 1, n ).
*           Unchanged on exit.
*
*  B      - DOUBLE PRECISION array of DIMENSION ( LDB, n ).
*           Before entry, the leading  m by n part of the array  B  must
*           contain the matrix B.
*           Unchanged on exit.
*
*  LDB    - INTEGER.
*           On entry, LDB specifies the first dimension of B as declared
*           in  the  calling  (sub)  program.   LDB  must  be  at  least
*           max( 1, m ).
*           Unchanged on exit.
*
*  BETA   - DOUBLE PRECISION.
*           On entry,  BETA  specifies the scalar  beta.  When  BETA  is
*           supplied as zero then C need not be set on input.
*           Unchanged on exit.
*
*  C      - DOUBLE PRECISION array of DIMENSION ( LDC, n ).
*           Before entry, the leading  m by n  part of the array  C must
*           contain the matrix  C,  except when  beta  is zero, in which
*           case C need not be set on entry.
*           On exit, the array  C  is overwritten by the  m by n updated
*           matrix.
*
*  LDC    - INTEGER.
*           On entry, LDC specifies the first dimension of C as declared
*           in  the  calling  (sub)  program.   LDC  must  be  at  least
*           max( 1, m ).
*           Unchanged on exit.
*
*
*  Level 3 Blas routine.
*
*  -- Written on 8-February-1989.
*     Jack Dongarra, Argonne National Laboratory.
*     Iain Duff, AERE Harwell.
*     Jeremy Du Croz, Numerical Algorithms Group Ltd.
*     Sven Hammarling, Numerical Algorithms Group Ltd.
*
*  -- Rewritten in December-1993.
*     GEMM-Based Level 3 BLAS.
*     Per Ling, Institute of Information Processing,
*     University of Umea, Sweden.
*
*
*     .. Local Scalars ..
      INTEGER            INFO, NROWA
      LOGICAL            LSIDE, UPPER
      INTEGER            I, II, IX, ISEC, J, JJ, JX, JSEC
*     .. Intrinsic Functions ..
      INTRINSIC          MAX, MIN
*     .. External Functions ..
      LOGICAL            LSAME
      EXTERNAL           LSAME
*     .. External Subroutines ..
      EXTERNAL           XERBLA
      EXTERNAL           DGEMM, DCOPY
*     .. Parameters ..
      DOUBLE PRECISION   ZERO, ONE
      PARAMETER        ( ZERO = 0.0D+0, ONE = 1.0D+0 )
*     .. User specified parameters for DSYMM ..
      INTEGER            RCB, CB
      PARAMETER        ( RCB = 128, CB = 64 )
*     .. Local Arrays ..
      DOUBLE PRECISION   T1( RCB, RCB )
*     ..
*     .. Executable Statements ..
*
*     Test the input parameters.
*
      LSIDE = LSAME( SIDE, 'L' )
      UPPER = LSAME( UPLO, 'U' )
      IF( LSIDE )THEN
         NROWA = M
      ELSE
         NROWA = N
      END IF
      INFO = 0
      IF( ( .NOT.LSIDE ).AND.( .NOT.LSAME( SIDE, 'R' ) ) )THEN
         INFO = 1
      ELSE IF( ( .NOT.UPPER ).AND.( .NOT.LSAME( UPLO, 'L' ) ) )THEN
         INFO = 2
      ELSE IF( M.LT.0 )THEN
         INFO = 3
      ELSE IF( N.LT.0 )THEN
         INFO = 4
      ELSE IF( LDA.LT.MAX( 1, NROWA ) )THEN
         INFO = 7
      ELSE IF( LDB.LT.MAX( 1, M ) )THEN
         INFO = 9
      ELSE IF( LDC.LT.MAX( 1, M ) )THEN
         INFO = 12
      END IF
      IF( INFO.NE.0 )THEN
         CALL XERBLA( 'DSYMM ', INFO )
         RETURN
      END IF
*
*     Quick return if possible.
*
      IF( ( M.EQ.0 ).OR.( N.EQ.0 ).OR.
     $    ( ( ALPHA.EQ.ZERO ).AND.( BETA.EQ.ONE ) ) )
     $   RETURN
*
*     And when alpha.eq.zero.
*
      IF( ALPHA.EQ.ZERO )THEN
         CALL DGEMM ( 'N', 'N', M, N, 0, ZERO, A, MAX( LDA, LDB ),
     $                                B, MAX( LDA, LDB ), BETA, C, LDC )
         RETURN
      END IF
*
*     Start the operations.
*
      IF( LSIDE )THEN
         IF( UPPER )THEN
*
*           Form  C := alpha*A*B + beta*C. Left, Upper.
*
            DO 40, II = 1, M, RCB
               ISEC = MIN( RCB, M-II+1 )
*
*              T1 := A, a upper triangular diagonal block of A is copied
*              to the upper triangular part of T1.
*
               DO 10, I = II, II+ISEC-1
                  CALL DCOPY ( I-II+1, A( II, I ), 1, T1( 1, I-II+1 ),
     $                                                               1 )
   10          CONTINUE
*
*              T1 :=  A', a strictly upper triangular diagonal block of
*              A is copied to the strictly lower triangular part of T1.
*              Notice that T1 is referenced by row and that the maximum
*              length of a vector referenced by DCOPY is CB.
*
               DO 30, JJ = II, II+ISEC-1, CB
                  JSEC = MIN( CB, II+ISEC-JJ )
                  DO 20, J = JJ+1, II+ISEC-1
                     CALL DCOPY ( MIN( JSEC, J-JJ ), A( JJ, J ), 1,
     $                                      T1( J-II+1, JJ-II+1 ), RCB )
   20             CONTINUE
   30          CONTINUE
*
*              C := alpha*T1*B + beta*C, a horizontal block of C is
*              updated using the general matrix multiply, DGEMM. T1
*              corresponds to a full diagonal block of the matrix A.
*
               CALL DGEMM ( 'N', 'N', ISEC, N, ISEC, ALPHA, T1( 1, 1 ),
     $                     RCB, B( II, 1 ), LDB, BETA, C( II, 1 ), LDC )
*
*              C := alpha*A'*B + C and C := alpha*A*B + C, general
*              matrix multiply operations involving rectangular blocks
*              of A.
*
               IF( II.GT.1 )THEN
                  CALL DGEMM ( 'T', 'N', ISEC, N, II-1, ALPHA,
     $                                  A( 1, II ), LDA, B( 1, 1 ), LDB,
     $                                            ONE, C( II, 1 ), LDC )
               END IF
               IF( II+ISEC.LE.M )THEN
                  CALL DGEMM ( 'N', 'N', ISEC, N, M-II-ISEC+1, ALPHA,
     $                           A( II, II+ISEC ), LDA, B( II+ISEC, 1 ),
     $                                       LDB, ONE, C( II, 1 ), LDC )
               END IF
   40       CONTINUE
         ELSE
*
*           Form  C := alpha*A*B + beta*C. Left, Lower.
*
            DO 80, IX = M, 1, -RCB
               II = MAX( 1, IX-RCB+1 )
               ISEC = IX-II+1
*
*              T1 := A, a lower triangular diagonal block of A is copied
*              to the lower triangular part of T1.
*
               DO 50, I = II, II+ISEC-1
                  CALL DCOPY ( II+ISEC-I, A( I, I ), 1,
     $                                         T1( I-II+1, I-II+1 ), 1 )
   50          CONTINUE
*
*              T1 :=  A', a strictly lower triangular diagonal block of
*              A is copied to the strictly upper triangular part of T1.
*              Notice that T1 is referenced by row and that the maximum
*              length of a vector referenced by DCOPY is CB.
*
               DO 70, JX = II+ISEC-1, II, -CB
                  JJ = MAX( II, JX-CB+1 )
                  JSEC = JX-JJ+1
                  DO 60, J = II, JJ+JSEC-2
                     CALL DCOPY ( MIN( JSEC, JJ+JSEC-1-J ),
     $                                        A( MAX( JJ, J+1 ), J ), 1,
     $                       T1( J-II+1, MAX( JJ-II+1, J-II+2 ) ), RCB )
   60             CONTINUE
   70          CONTINUE
*
*              C := alpha*T1*B + beta*C, a horizontal block of C is
*              updated using the general matrix multiply, DGEMM. T1
*              corresponds to a full diagonal block of the matrix A.
*
               CALL DGEMM ( 'N', 'N', ISEC, N, ISEC, ALPHA, T1( 1, 1 ),
     $                     RCB, B( II, 1 ), LDB, BETA, C( II, 1 ), LDC )
*
*              C := alpha*A'*B + C and C := alpha*A*B + C, general
*              matrix multiply operations involving rectangular blocks
*              of A.
*
               IF( II+ISEC.LE.M )THEN
                  CALL DGEMM ( 'T', 'N', ISEC, N, M-II-ISEC+1, ALPHA,
     $                           A( II+ISEC, II ), LDA, B( II+ISEC, 1 ),
     $                                       LDB, ONE, C( II, 1 ), LDC )
               END IF
               IF( II.GT.1 )THEN
                  CALL DGEMM ( 'N', 'N', ISEC, N, II-1, ALPHA,
     $                                  A( II, 1 ), LDA, B( 1, 1 ), LDB,
     $                                            ONE, C( II, 1 ), LDC )
               END IF
   80       CONTINUE
         END IF
      ELSE
         IF( UPPER )THEN
*
*           Form  C := alpha*B*A + beta*C. Right, Upper.
*
            DO 120, JJ = 1, N, RCB
               JSEC = MIN( RCB, N-JJ+1 )
*
*              T1 := A, a upper triangular diagonal block of A is copied
*              to the upper triangular part of T1.
*
               DO 90, J = JJ, JJ+JSEC-1
                  CALL DCOPY ( J-JJ+1, A( JJ, J ), 1, T1( 1, J-JJ+1 ),
     $                                                               1 )
   90          CONTINUE
*
*              T1 :=  A', a strictly upper triangular diagonal block of
*              A is copied to the strictly lower triangular part of T1.
*              Notice that T1 is referenced by row and that the maximum
*              length of a vector referenced by DCOPY is CB.
*
               DO 110, II = JJ, JJ+JSEC-1, CB
                  ISEC = MIN( CB, JJ+JSEC-II )
                  DO 100, I = II+1, JJ+JSEC-1
                     CALL DCOPY ( MIN( ISEC, I-II ), A( II, I ), 1,
     $                                      T1( I-JJ+1, II-JJ+1 ), RCB )
  100             CONTINUE
  110          CONTINUE
*
*              C := alpha*B*T1 + beta*C, a vertical block of C is updated
*              using the general matrix multiply, DGEMM. T1 corresponds
*              to a full diagonal block of the matrix A.
*
               CALL DGEMM ( 'N', 'N', M, JSEC, JSEC, ALPHA, B( 1, JJ ),
     $                     LDB, T1( 1, 1 ), RCB, BETA, C( 1, JJ ), LDC )
*
*              C := alpha*B*A + C and C := alpha*B*A' + C, general
*              matrix multiply operations involving rectangular blocks
*              of A.
*
               IF( JJ.GT.1 )THEN
                  CALL DGEMM ( 'N', 'N', M, JSEC, JJ-1, ALPHA,
     $                                  B( 1, 1 ), LDB, A( 1, JJ ), LDA,
     $                                            ONE, C( 1, JJ ), LDC )
               END IF
               IF( JJ+JSEC.LE.N )THEN
                  CALL DGEMM ( 'N', 'T', M, JSEC, N-JJ-JSEC+1, ALPHA,
     $                           B( 1, JJ+JSEC ), LDB, A( JJ, JJ+JSEC ),
     $                                       LDA, ONE, C( 1, JJ ), LDC )
               END IF
  120       CONTINUE
         ELSE
*
*           Form  C := alpha*B*A + beta*C. Right, Lower.
*
            DO 160, JX = N, 1, -RCB
               JJ = MAX( 1, JX-RCB+1 )
               JSEC = JX-JJ+1
*
*              T1 := A, a lower triangular diagonal block of A is copied
*              to the lower triangular part of T1.
*
               DO 130, J = JJ, JJ+JSEC-1
                  CALL DCOPY ( JJ+JSEC-J, A( J, J ), 1,
     $                                         T1( J-JJ+1, J-JJ+1 ), 1 )
  130          CONTINUE
*
*              T1 :=  A', a strictly lower triangular diagonal block of
*              A is copied to the strictly upper triangular part of T1.
*              Notice that T1 is referenced by row and that the maximum
*              length of a vector referenced by DCOPY is CB.
*
               DO 150, IX = JJ+JSEC-1, JJ, -CB
                  II = MAX( JJ, IX-CB+1 )
                  ISEC = IX-II+1
                  DO 140, I = JJ, II+ISEC-2
                     CALL DCOPY ( MIN( ISEC, II+ISEC-1-I ),
     $                                        A( MAX( II, I+1 ), I ), 1,
     $                       T1( I-JJ+1, MAX( II-JJ+1, I-JJ+2 ) ), RCB )
  140             CONTINUE
  150          CONTINUE
*
*              C := alpha*B*T1 + beta*C, a vertical block of C is
*              updated using the general matrix multiply, DGEMM. T1
*              corresponds to a full diagonal block of the matrix A.
*
               CALL DGEMM ( 'N', 'N', M, JSEC, JSEC, ALPHA,
     $                                 B( 1, JJ ), LDB, T1( 1, 1 ), RCB,
     $                                           BETA, C( 1, JJ ), LDC )
*
*              C := alpha*B*A + C and C := alpha*B*A' + C, general
*              matrix multiply operations involving rectangular blocks
*              of A.
*
               IF( JJ+JSEC.LE.N )THEN
                  CALL DGEMM ( 'N', 'N', M, JSEC, N-JJ-JSEC+1, ALPHA,
     $                           B( 1, JJ+JSEC ), LDB, A( JJ+JSEC, JJ ),
     $                                       LDA, ONE, C( 1, JJ ), LDC )
               END IF
               IF( JJ.GT.1 )THEN
                  CALL DGEMM ( 'N', 'T', M, JSEC, JJ-1, ALPHA,
     $                                  B( 1, 1 ), LDB, A( JJ, 1 ), LDA,
     $                                            ONE, C( 1, JJ ), LDC )
               END IF
  160       CONTINUE
         END IF
      END IF
*
      RETURN
*
*     End of DSYMM.
*
      END
*/
