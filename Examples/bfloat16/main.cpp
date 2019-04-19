#include <iostream>
#include <cstdio>
#include <math.h>
#include <iomanip>
#include <limits>
#include "rocblas_bfloat16.h"
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>


int main()
{
//  test multiply add
    rocblas_bfloat16 b0 = rocblas_bfloat16(0.0f); float f0 = 0.0f;
    rocblas_bfloat16 b1 = rocblas_bfloat16(1.0f); float f1 = 1.0f;
    rocblas_bfloat16 b2 = rocblas_bfloat16(2.0f); float f2 = 2.0f;
    rocblas_bfloat16 b3 = rocblas_bfloat16(3.0f); float f3 = 3.0f;
    rocblas_bfloat16 b4 = rocblas_bfloat16(4.0f); float f4 = 4.0f;
    rocblas_bfloat16 b5 = rocblas_bfloat16(5.0f); float f5 = 5.0f;
    rocblas_bfloat16 b6 = rocblas_bfloat16(6.0f); float f6 = 6.0f;
    rocblas_bfloat16 b7 = rocblas_bfloat16(7.0f); float f7 = 7.0f;
    rocblas_bfloat16 b8 = rocblas_bfloat16(8.0f); float f8 = 8.0f;
    rocblas_bfloat16 b9 = rocblas_bfloat16(9.0f); float f9 = 9.0f;
    rocblas_bfloat16 b1_1 = rocblas_bfloat16(1.1f); float f1_1 = 1.1f;

    rocblas_bfloat16 bb = b2 * ((b3 * b4) + (b5 * b6)) + b7 * b8;
    float            ff = f2 * ((f3 * f4) + (f5 * f6)) + f7 * f8;

    std::cout << "calculated bb = a2 * ((a3 * b4) + (a5 * b6)) + a7 * a8 = " << bb << std::endl;
    std::cout << "reference  ff = f2 * ((f3 * f4) + (f5 * f6)) + f7 * f8 = " << ff << std::endl << std::endl;

    bb = b1_1 * ((b3 * b4) + (b5 * b6)) + b7 * b8;
    ff = f1_1 * ((f3 * f4) + (f5 * f6)) + f7 * f8;

    std::cout << "calculated bb = a1_1 * ((a3 * b4) + (a5 * b6)) + a7 * a8 = " << bb << std::endl;
    std::cout << "reference  ff = f1_1 * ((f3 * f4) + (f5 * f6)) + f7 * f8 = " << ff << std::endl;

    return 0;
}
