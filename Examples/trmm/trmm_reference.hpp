      template<typename T> rocblas_status trmm_reference (
              rocblas_side side, 
              rocblas_fill uplo, 
              rocblas_operation trans, 
              rocblas_diagonal diag, 
              rocblas_int m, 
              rocblas_int n, 
              T alpha, 
              T *a, 
              rocblas_int lda, 
              T *b, 
              rocblas_int ldb)
{
//
// -- Reference BLAS level3 routine (version 3.7.0) --
// -- Reference BLAS is a software package provided by Univ. of Tennessee,    --
// -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..--
//    December 2016
//
//    .. Local Scalars ..
      T temp;
      rocblas_int nrowa;
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
//
      if (m < 0 || n < 0 || lda < nrowa || ldb < m)
      {
          return rocblas_status_invalid_size;
      }
//
//    Quick return if possible.
//
      if (m == 0 || n == 0) return rocblas_status_success;
//
//    And when  alpha.eq.zero.
//
      if (alpha == 0.0)
      {
          for (int j = 0; j < n; j++)
          {
              for (int i = 0; i < m; i++)
              {
                  b[i + j*ldb] = 0.0;
              }
          }
      }
//
//    Start the operations.
//
      if (side == rocblas_side_left)
      {
          if(trans == rocblas_operation_none)
          {
//
//          Form  B := alpha*A*B.
//
              if (uplo == rocblas_fill_upper)
              {
                   for(int j = 1; j <= n; j++)
                   {
                       for (int k = 1; k <= m; k++)
                       {
                          if (b[k-1 + (j-1)*ldb] != 0.0)
                          {
                              temp = alpha*b[k-1 + (j-1)*ldb];
                               for (int i = 1; i <= k-1; i++)
                               {
                                  b[i-1 + (j-1)*ldb] = b[i-1 + (j-1)*ldb] + temp * a[i-1 + (k-1)*lda];
                               }
                               if (diag == rocblas_diagonal_non_unit)
                               {
                                   temp = temp*a[k-1 + (k-1)*lda];
                               }
                               b[k-1 + (j-1)*ldb] = temp;
                          }
                      }
                  }
              }
              else
              {
                  for (int j = 1; j <= n; j++)
                  {
                       for (int k = m; k >= 1; k--)
                       {
                          if (b[k-1 + (j-1)*ldb] != 0.0)
                          {
                              temp = alpha*b[k-1 + (j-1)*ldb];
                              b[k-1 + (j-1)*ldb] = temp;
                              if (diag == rocblas_diagonal_non_unit)
                              { 
                                  b[k-1 + (j-1)*ldb] = b[k-1 + (j-1)*ldb]*a[k-1 + (k-1)*lda];
                              }
                              for (int i = k + 1; i <= m; i++)
                              {
                                  b[i-1 + (j-1)*ldb] = b[i-1 + (j-1)*ldb] + temp*a[i-1 + (k-1)*lda];
                              }
                          }
                      }
                  }
              }
          }
          else
          {
//
//          Form  B := alpha*A**T*B.
//
              if (uplo == rocblas_fill_upper)
              {
                   for (int j = 1; j <= n; j++)
                   {
                       for (int i = m; i >= 1; i--)
                       {
                          temp = b[i-1 + (j-1)*ldb];
                          if (diag == rocblas_diagonal_non_unit)
                          {
                              temp = temp*a[i-1 + (i-1)*lda];
                          }
                          for (int k = 1; k <= i - 1; k++)
                          {
                              temp = temp + a[k-1 + (i-1)*lda]*b[k-1 + (j-1)*ldb];
                          }
                          b[i-1 + (j-1)*ldb] = alpha*temp;
                      }
                  }
              }
              else
              {
                   for (int j = 1; j <= n; j++)
                   {
                       for (int i = 1; i <= m; i++)
                       {
                          temp = b[i-1 + (j-1)*ldb];
                          if (diag == rocblas_diagonal_non_unit)
                          {
                              temp = temp*a[i-1 + (i-1)*lda];
                          }
                          for (int k = i + 1; k <= m; k++)
                          {
                              temp = temp + a[k-1 + (i-1)*lda]*b[k-1 + (j-1)*ldb];
                          }
                          b[i-1 + (j-1)*ldb] = alpha * temp;
                      }
                  }
              }
          }
      }
      else
      {
          if (trans == rocblas_operation_none)
          {
//
//          Form  B := alpha*B*A.
//
              if (uplo == rocblas_fill_upper)
              {
                   for (int j = n; j >= 1; j--)
                   {
                      temp = alpha;
                      if (diag == rocblas_diagonal_non_unit)
                      {
                          temp = temp*a[j-1 + (j-1)*lda];
                      }
                      for (int i = 1; i <= m; i++)
                      {
                          b[i-1 + (j-1)*ldb] = temp*b[i-1 + (j-1)*ldb];
                      }
                      for (int k = 1; k <= j - 1; k++)
                      {
                          if (a[k-1 + (j-1)*lda] != 0.0)
                          {
                              temp = alpha*a[k-1 + (j-1)*lda];
                              for (int i = 1; i <= m; i++)
                              {
                                  b[i-1 + (j-1)*ldb] = b[i-1 + (j-1)*ldb] + temp*b[i-1 + (k-1)*ldb];
                              }
                          }
                      }
                  }
              }
              else
              {
                   for (int j = 1; j <= n; j++)
                   {
                      temp = alpha;
                      if (diag == rocblas_diagonal_non_unit)
                      {
                          temp = temp*a[j-1 + (j-1)*lda];
                      }
                      for (int i = 1; i <= m; i++)
                      {
                          b[i-1 + (j-1)*ldb] = temp*b[i-1 + (j-1)*ldb];
                      }
                      for (int k = j + 1; k <= n; k++)
                      {
                          if (a[k-1 + (j-1)*lda] != 0.0)
                          {
                              temp = alpha*a[k-1 + (j-1)*lda];
                              for (int i = 1; i <= m; i++)
                              {
                                  b[i-1 + (j-1)*ldb] = b[i-1 + (j-1)*ldb] + temp*b[i-1 + (k-1)*ldb];
                              }
                          }
                      }
                  }
              }
          }
          else
          {
//
//          Form  B := alpha*B*A**T.
//
              if (uplo == rocblas_fill_upper)
              {
                   for (int k = 1; k <= n; k++)
                   {
                       for (int j = 1; j <= k - 1; j++)
                       {
                          if (a[j-1 + (k-1)*lda] != 0.0)
                          {
                              temp = alpha*a[j-1 + (k-1)*lda];
                              for (int i = 1; i <= m; i++)
                              {
                                  b[i-1 + (j-1)*ldb] = b[i-1 + (j-1)*ldb] + temp*b[i-1 + (k-1)*ldb];
                              }
                          }
                      }
                      temp = alpha;
                      if (diag == rocblas_diagonal_non_unit)
                      {
                          temp = temp*a[k-1 + (k-1)*lda];
                      }
                      if (temp != 1)
                      {
                          for (int i = 1; i <= m; i++)
                          {
                              b[i-1 + (k-1)*ldb] = temp*b[i-1 + (k-1)*ldb];
                          }
                      }
                  }
              }
              else
              {
                   for (int k = n; k >= 1; k--)
                   {
                       for (int j = k + 1; j <= n; j++)
                       {
                          if (a[j-1 + (k-1)*lda] != 0.0)
                          {
                              temp = alpha*a[j-1 + (k-1)*lda];
                               for (int i = 1; i <= m; i++)
                               {
                                  b[i-1 + (j-1)*ldb] = b[i-1 + (j-1)*ldb] + temp*b[i-1 + (k-1)*ldb];
                              }
                          }
                      }
                      temp = alpha;
                      if (diag == rocblas_diagonal_non_unit)
                      {
                          temp = temp*a[k-1 + (k-1)*lda];
                      }
                      if (temp != 1)
                      {
                          for (int i = 1; i <=m; i++)
                          {
                              b[i-1 + (k-1)*ldb] = temp*b[i-1 + (k-1)*ldb];
                          }
                      }
                  }
              }
          }
      }

      return rocblas_status_success;
}
