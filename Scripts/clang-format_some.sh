#!/bin/bash

FILES="
./blas3/trtri.hpp
./blas3/trtri_trsm.hpp
./blas3/trtri_batched.hpp
./blas3/rocblas_trmm.cpp
./rocblas_auxiliary.cpp
./blas3/rocblas_geam.cpp
./blas2/rocblas_ger.cpp
./blas2/rocblas_gemv.cpp
./blas1/rocblas_amin.cpp
./blas1/rocblas_dot.cpp
./blas1/rocblas_axpy.cpp
./blas1/rocblas_amax.cpp
./blas1/rocblas_swap.cpp
./blas1/rocblas_copy.cpp
./blas1/rocblas_asum.cpp
./blas1/rocblas_scal.cpp
./blas1/rocblas_nrm2.cpp
"

for F in $FILES 
do
   echo "filename = $F"
   clang-format-3.8 -i -style=file $F
done
