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
//  rocblas_bfloat16 b0 = static_cast<rocblas_bfloat16>(0.0f); float f0 = 0.0f;
    rocblas_bfloat16 b0 = static_cast<rocblas_bfloat16>(0.0f); float f0 = 0.0f;
    rocblas_bfloat16 b1 = static_cast<rocblas_bfloat16>(1.0f); float f1 = 1.0f;
    rocblas_bfloat16 b2 = static_cast<rocblas_bfloat16>(2.0f); float f2 = 2.0f;
    rocblas_bfloat16 b3 = static_cast<rocblas_bfloat16>(3.0f); float f3 = 3.0f;
    rocblas_bfloat16 b4 = static_cast<rocblas_bfloat16>(4.0f); float f4 = 4.0f;
    rocblas_bfloat16 b5 = static_cast<rocblas_bfloat16>(5.0f); float f5 = 5.0f;
    rocblas_bfloat16 b6 = static_cast<rocblas_bfloat16>(6.0f); float f6 = 6.0f;
    rocblas_bfloat16 b7 = static_cast<rocblas_bfloat16>(7.0f); float f7 = 7.0f;
    rocblas_bfloat16 b8 = static_cast<rocblas_bfloat16>(8.0f); float f8 = 8.0f;
    rocblas_bfloat16 b9 = static_cast<rocblas_bfloat16>(9.0f); float f9 = 9.0f;
    rocblas_bfloat16 b1_1 = static_cast<rocblas_bfloat16>(1.1f); float f1_1 = 1.1f;

    double d3 = 3.0; int i3 = 3;
    double d5 = 5.0; int i5 = 5;

    rocblas_bfloat16 bb = b2 * ((b3 * b4) + (b5 * b6)) - b7 * b8;
    float            ff = f2 * ((f3 * f4) + (f5 * f6)) - f7 * f8;

    std::cout << "size of rocblas_bfloat16 = " << sizeof(b0) << " bytes" << std::endl;
    std::cout << "size of float            = " << sizeof(ff) << " bytes" << std::endl << std::endl;

    std::cout << "test operators *, +, - " << std::endl;
    std::cout << "calculated bb = b2 * ((b3 * b4) + (b5 * b6)) - b7 * b8 = " << bb << std::endl;
    std::cout << "reference  ff = f2 * ((f3 * f4) + (f5 * f6)) - f7 * f8 = " << ff << std::endl << std::endl;

    std::cout << "calculated b1_1 * b3 = " << b1_1 * b3 << std::endl;
    std::cout << "reference  f1_1 * f3 = " << f1_1 * f3 << std::endl;

    std::cout << "test operator / " << std::endl;
    std::cout << "calculated b8 / b2 = " << b8 / b2 << std::endl;
    std::cout << "reference  f8 / f2 = " << f8 / f2 << std::endl;

    std::cout << "test operators: >, <, ==, !=, <=, >=" << std::endl;
    b2 > b1 ? std::cout << "PASS: b2 > b1\n" : std::cout << "FAIL: b2 > b1\n";
    b1 < b2 ? std::cout << "PASS: b1 < b2\n" : std::cout << "FAIL: b1 > b2\n";

    b1 == b1 ? std::cout << "PASS: b1 == b2\n" : std::cout << "FAIL: b1 == b2\n";
    b1 == b2 ? std::cout << "FAIL: b1 == b2\n" : std::cout << "PASS: b1 == b2\n";

    b1 != b1 ? std::cout << "FAIL: b1 != b1\n" : std::cout << "PASS: b1 != b1\n";
    b1 != b2 ? std::cout << "PASS: b1 != b2\n" : std::cout << "FAIL: b1 != b2\n";

    b1 >= b1 ? std::cout << "PASS: b1 >= b1\n" : std::cout << "FAIL: b1 >= b1\n";
    b2 >= b1 ? std::cout << "PASS: b2 >= b1\n" : std::cout << "FAIL: b2 >= b1\n";
    b1 >= b2 ? std::cout << "FAIL: b1 >= b2\n" : std::cout << "PASS: b1 >= b2\n"; 

    b1 <= b1 ? std::cout << "PASS: b1 <= b1\n" : std::cout << "FAIL: b1 <= b1\n";
    b2 <= b1 ? std::cout << "FAIL: b2 <= b1\n" : std::cout << "PASS: b2 <= b1\n";
    b1 <= b2 ? std::cout << "PASS: b1 <= b2\n" : std::cout << "FAIL: b1 <= b2\n";

    rocblas_bfloat16 tt;
    tt = b2; (tt += b1) == b3 ? std::cout << "PASS: (b2 += b1) == b3\n" : std::cout << "FAIL: (b2 += b1) == b3\n";
    tt = b2; (tt -= b1) == b1 ? std::cout << "PASS: (b2 -= b1) == b1\n" : std::cout << "FAIL: (b2 -= b1) == b1\n";
    tt = b2; (tt *= b3) == b6 ? std::cout << "PASS: (b2 *= b3) == b6\n" : std::cout << "FAIL: (b2 *= b3) == b6\n";
    tt = b6; (tt /= b2) == b3 ? std::cout << "PASS: (b6 /= b2) == b3\n" : std::cout << "FAIL: (b6 /= b2) == b3\n";

    tt = b2 / b0; if(isinf(tt)) {std::cout << "PASS: isinf\n";} else { std::cout << "FAIL: isinf\n";};
    tt = tt / tt; if(isnan(tt)) {std::cout << "PASS: isnan\n";} else { std::cout << "FAIL: isnan\n";};

    b8 == abs(b9 - b1) ? std::cout << "PASS: abs\n" : std::cout << "FAIL: abs\n";
    b9 == abs(b7 + b1) ? std::cout << "FAIL: abs\n" : std::cout << "PASS: abs\n";

    sin(b0) == b0 ? std::cout << "PASS: sin(b0) == 0\n" : std::cout << "FAIL: sin(b0) == 0\n";
    cos(b0) == b1 ? std::cout << "PASS: cos(b0) == b1\n" : std::cout << "FAIL: cos(b0) == b1\n";

    rocblas_bfloat16 temp = static_cast<rocblas_bfloat16>(d3);
    temp == b3 ? std::cout << "PASS: " : std::cout << "FAIL: ";
    std::cout << "static_cast<rocblas_bfloat16>(d3) = " << temp << std::endl;

    temp = static_cast<rocblas_bfloat16>(i5);
    temp == b5 ? std::cout << "PASS: " : std::cout << "FAIL: ";
    std::cout << "static_cast<rocblas_bfloat16>(d5) = " << temp << std::endl;



    return 0;
}
