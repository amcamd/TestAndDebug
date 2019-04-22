#include <cmath>
#include <cinttypes>
#include <iostream>

#ifndef __BYTE_ORDER__
#define __BYTE_ORDER__ __ORDER_LITTLE_ENDIAN__
#endif

#define BFLOAT16_Q_NAN_VALUE 0xFFC1

typedef struct
{
    uint16_t data;
} rocblas_bfloat16;

// zero extend lower 16 bits of bfloat16 to convert to IEEE float
float bfloat16_to_float(const rocblas_bfloat16 v)
{
    union
    {
        float fp32 = 0;
        uint16_t q[2];
    };

#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
    q[0] = v.data;
#else
    q[1] = v.data;
#endif
    return fp32;
}

// truncate lower 16 bits of IEEE float to convert to bfloat16
rocblas_bfloat16 float_to_bfloat16(const float v)
{
    rocblas_bfloat16 bf16;
    if (std::isnan(v))
    {
        bf16.data = BFLOAT16_Q_NAN_VALUE;
        return bf16;
    }
    union {
        float fp32;
        uint16_t p[2];
    };
    fp32 = v;

#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
    bf16.data = p[0];
#else
    bf16.data = p[1];
#endif
    return bf16;
}

inline std::ostream& operator<<(std::ostream& os, const rocblas_bfloat16& bf16) { os << bfloat16_to_float(bf16); return os; }

inline rocblas_bfloat16 operator+(rocblas_bfloat16 a, rocblas_bfloat16 b) { return float_to_bfloat16(bfloat16_to_float(a) + bfloat16_to_float(b)); }
inline rocblas_bfloat16 operator-(rocblas_bfloat16 a, rocblas_bfloat16 b) { return float_to_bfloat16(bfloat16_to_float(a) - bfloat16_to_float(b)); }
inline rocblas_bfloat16 operator*(rocblas_bfloat16 a, rocblas_bfloat16 b) { return float_to_bfloat16(bfloat16_to_float(a) * bfloat16_to_float(b)); }
inline rocblas_bfloat16 operator/(rocblas_bfloat16 a, rocblas_bfloat16 b) { return float_to_bfloat16(bfloat16_to_float(a) / bfloat16_to_float(b)); }

inline bool operator<(rocblas_bfloat16 a, rocblas_bfloat16 b) { return bfloat16_to_float(a) < bfloat16_to_float(b); }
inline bool operator<=(rocblas_bfloat16 a, rocblas_bfloat16 b) { return bfloat16_to_float(a) <= bfloat16_to_float(b); }
inline bool operator==(rocblas_bfloat16 a, rocblas_bfloat16 b) { return bfloat16_to_float(a) == bfloat16_to_float(b); }
inline bool operator!=(rocblas_bfloat16 a, rocblas_bfloat16 b) { return bfloat16_to_float(a) != bfloat16_to_float(b); }
inline bool operator>(rocblas_bfloat16 a, rocblas_bfloat16 b) { return bfloat16_to_float(a) > bfloat16_to_float(b); }
inline bool operator>=(rocblas_bfloat16 a, rocblas_bfloat16 b) { return bfloat16_to_float(a) >= bfloat16_to_float(b); }

inline rocblas_bfloat16& operator+=(rocblas_bfloat16& a, rocblas_bfloat16 b) { a = a + b; return a; }
inline rocblas_bfloat16& operator-=(rocblas_bfloat16& a, rocblas_bfloat16 b) { a = a - b; return a; }
inline rocblas_bfloat16& operator*=(rocblas_bfloat16& a, rocblas_bfloat16 b) { a = a * b; return a; }
inline rocblas_bfloat16& operator/=(rocblas_bfloat16& a, rocblas_bfloat16 b) { a = a / b; return a; }

inline bool isinf(const rocblas_bfloat16& a) { return std::isinf(bfloat16_to_float(a)); }
inline bool isnan(const rocblas_bfloat16& a) { return std::isnan(bfloat16_to_float(a)); }
inline bool iszero(const rocblas_bfloat16& a) { return (a.data & 0x7FFF) == 0; }

inline rocblas_bfloat16 abs(const rocblas_bfloat16& a) { return float_to_bfloat16(std::abs(bfloat16_to_float(a))); }
inline rocblas_bfloat16 sin(const rocblas_bfloat16& a) { return float_to_bfloat16(std::sin(bfloat16_to_float(a))); }
inline rocblas_bfloat16 cos(const rocblas_bfloat16& a) { return float_to_bfloat16(std::cos(bfloat16_to_float(a))); }
