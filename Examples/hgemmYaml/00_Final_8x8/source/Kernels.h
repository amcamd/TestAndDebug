#pragma once
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
// Header

extern "C"
__global__ void Cij_Aik_Bkj_HB_MT128x128x08_K1(
  hipLaunchParm lp,
  half *C,
  half const * __restrict__ A,
  half const * __restrict__ B,
  half const alpha,
  half const beta,
  unsigned int const offsetC,
  unsigned int const offsetA,
  unsigned int const offsetB,
  unsigned int const strideC1J,
  unsigned int const strideAK,
  unsigned int const strideB1J,
  unsigned int const size0I,
  unsigned int const size1J,
  unsigned int const sizeK );
