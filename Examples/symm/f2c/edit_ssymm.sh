#!/bin/bash

FILES="
./ssymm.cpp
"

for F in $FILES 
do
   echo "filename = $F"
   cp "$F" "$F.bak"
   sed -e 's:void SSYMM\(side, uplo, M, N, alpha, A, LDA, B, LDB, beta, C, LDC\):void SSYMM\( rocblas_side side, rocblas_fill uplo, rocblas_int M, rocblas_int N, T alpha, T *A, rocblas_int lda, T *B, rocblas_int ldb, T beta, T *C, rocblas_int ldc\):' \
       -e 's:^    EXTERNAL:    //EXTERNAL:g' \
       -e 's:^    INTRINSIC:    //INTRINSIC:g' \
       -e 's:^    const:    :g' \
       -e 's:E + 0::g' \
       -e 's:BETA:beta:g' \
       -e 's:ALPHA:alpha:g' \
       -e 's:SIDE:side:g' \
       -e 's:UPLO:uplo:g' \
       -e 's:NROWA = M:nrowa = m:g' \
       -e 's:NROWA = N:nrowa= n:g' \
       -e 's:NROWA:nrowa:g' \
       -e 's:TEMP1:temp1:g' \
       -e 's:TEMP2:temp2:g' \
       -e 's:LDA:lda:g' \
       -e 's:LDB:ldb:g' \
       -e 's:LDC:ldc:g' \
       -e 's:::g' \
       -e 's:A\[I, K\]:a\[i + k*lda\]:g' \
       -e 's:A\[J, K\]:a\[j + k*lda\]:g' \
       -e 's:A\[J, J\]:a\[j + j*lda\]:g' \
       -e 's:A\[I, I\]:a\[i + i*lda\]:g' \
       -e 's:A\[K, I\]:a\[k + i*lda\]:g' \
       -e 's:A\[K, J\]:a\[k + j*lda\]:g' \
       -e 's:C\[I, J\]:c\[i + j*ldc\]:g' \
       -e 's:C\[K, J\]:c\[k + j*ldc\]:g' \
       -e 's:B\[I, J\]:b\[i + j*ldb\]:g' \
       -e 's:B\[K, J\]:b\[k + j*ldb\]:g' \
       -e 's:B\[I, K\]:b\[i + k*ldb\]:g' \
       -e 's:for\(J=1; J<=N; J++\):for\(int j=0; j<n; j++\):g' \
       -e 's:for(I=1; I<=M; I++):for(int i = 0; i < m; i++):g' \
       -e 's:} else:}\n    else:g' \
       -e 's:for(140 K = 1, J - 1)\\n:for (int k=0; k < j-1; k++)\n{:g' \
       -e 's:for(80 K = I + 1, M)\\n{:for (int k = i; k < m; k++)\n{:g' \
       -e 's:for(50 K = 1, I - 1)\\n{:for (int k = 0; k < i-1; k++)\n{:g' \
       -e 's:for(90 I = M, 1, -1)\\n{:for (int i = m-1; i>=0; i--)\n{:g' \
       -e 's:for(160 K = J + 1, N)\\n{:for(int k = j; k < n; k++)\n{:g' \
       "$F.bak" > "$F"
done


#      -e 's:::g' \


#include "rocblas-types.h"
#
#template<typename T>
#void SSYMM(
#     rocblas_side      side,   
#     rocblas_fill      uplo,
#     rocblas_int       m,
#     rocblas_int       n,
#     T                 alpha,
#     T                 *a,
#     rocblas_int       lda,
#     T                 *b,
#     rocblas_int       ldb,
#     T                 beta,
#     T                 *c,
#     rocblas_int       ldc)
#{
