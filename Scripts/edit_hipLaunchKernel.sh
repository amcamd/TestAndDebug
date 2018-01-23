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
   cp "$F" "$F.bak"
   sed -e 's/hipLaunchKernel/hipLaunchKernelGGL/g'  \
       -e 's/hipLaunchKernelGGLGGL/hipLaunchKernelGGL/g'  \
       -e 's/\(HIP_KERNEL_NAME(\)\(\w*\)\()\)/\2/g' \
       -e 's/\(HIP_KERNEL_NAME(\)\(\w*\)\(<.*>\)\()\)/(\2\3)/g' \
       -e 's/hipLaunchParm lp, //g' \
       -e 's/hipLaunchParm lp,//g' "$F.bak" > "$F"
done

# replace
# hipLaunchKernel(HIP_KERNEL_NAME(axpy_kernel_device_scalar),
# with
# hipLaunchKernelGGL(axpy_kernel_device_scalar,

# replace
# hipLaunchKernel(HIP_KERNEL_NAME(iamax_kernel_part1<T1, T2, NB_X>),
# with
# hipLaunchKernelGGL((iamax_kernel_part1<T1, T2, NB_X>),

# remove "hipLaunchParm lp," and "hipLaunchParm lp, "

