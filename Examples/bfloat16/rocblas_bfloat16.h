#include <cmath>
#include <cinttypes>
#include <iostream>

#ifndef __BYTE_ORDER__
#define __BYTE_ORDER__ __ORDER_LITTLE_ENDIAN__
#endif

#define BFLOAT16_Q_NAN_VALUE 0xFFC1

typedef struct rocblas_bfloat16
{
    rocblas_bfloat16(): data(BFLOAT16_ZERO_VALUE) {}
    // zero extend lower 16 bits of bfloat16 to convert to IEEE float
    static float bfloat16_to_float(const rocblas_bfloat16 v)
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
    static rocblas_bfloat16 float_to_bfloat16(const float v)
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

    explicit rocblas_bfloat16(const float v)
    {
        data = float_to_bfloat16(v).data;
    }

    explicit rocblas_bfloat16(const double v) { data = float_to_bfloat16(static_cast<float>(v)).data; }
    explicit rocblas_bfloat16(const int v) { data = float_to_bfloat16(static_cast<float>(v)).data; }

    explicit operator float() const
    {
        return bfloat16_to_float(*this);
    }

    uint16_t data;

    private:

        static const int16_t BFLOAT16_ZERO_VALUE = 0x00; 

} rocblas_bfloat16;

inline std::ostream& operator<<(std::ostream& os, const rocblas_bfloat16& bf16) { os << static_cast<float>(bf16); return os; }

inline rocblas_bfloat16 operator+(rocblas_bfloat16 a, rocblas_bfloat16 b) { return static_cast<rocblas_bfloat16>(static_cast<float>(a) + static_cast<float>(b)); }
inline rocblas_bfloat16 operator-(rocblas_bfloat16 a, rocblas_bfloat16 b) { return static_cast<rocblas_bfloat16>(static_cast<float>(a) - static_cast<float>(b)); }
inline rocblas_bfloat16 operator*(rocblas_bfloat16 a, rocblas_bfloat16 b) { return static_cast<rocblas_bfloat16>(static_cast<float>(a) * static_cast<float>(b)); }
inline rocblas_bfloat16 operator/(rocblas_bfloat16 a, rocblas_bfloat16 b) { return static_cast<rocblas_bfloat16>(static_cast<float>(a) / static_cast<float>(b)); }

inline bool operator<(rocblas_bfloat16 a, rocblas_bfloat16 b) { return static_cast<float>(a) < static_cast<float>(b); }
inline bool operator<=(rocblas_bfloat16 a, rocblas_bfloat16 b) { return static_cast<float>(a) <= static_cast<float>(b); }
inline bool operator==(rocblas_bfloat16 a, rocblas_bfloat16 b) { return static_cast<float>(a) == static_cast<float>(b); }
inline bool operator!=(rocblas_bfloat16 a, rocblas_bfloat16 b) { return static_cast<float>(a) != static_cast<float>(b); }
inline bool operator>(rocblas_bfloat16 a, rocblas_bfloat16 b) { return static_cast<float>(a) > static_cast<float>(b); }
inline bool operator>=(rocblas_bfloat16 a, rocblas_bfloat16 b) { return static_cast<float>(a) >= static_cast<float>(b); }

inline rocblas_bfloat16& operator+=(rocblas_bfloat16& a, rocblas_bfloat16 b) { a = a + b; return a; }
inline rocblas_bfloat16& operator-=(rocblas_bfloat16& a, rocblas_bfloat16 b) { a = a - b; return a; }
inline rocblas_bfloat16& operator*=(rocblas_bfloat16& a, rocblas_bfloat16 b) { a = a * b; return a; }
inline rocblas_bfloat16& operator/=(rocblas_bfloat16& a, rocblas_bfloat16 b) { a = a / b; return a; }

inline bool isinf(const rocblas_bfloat16& a) { return std::isinf(static_cast<float>(a)); }
inline bool isnan(const rocblas_bfloat16& a) { return std::isnan(static_cast<float>(a)); }
inline bool iszero(const rocblas_bfloat16& a) { return (a.data & 0x7FFF) == 0; }

inline rocblas_bfloat16 abs(const rocblas_bfloat16& a) { return static_cast<rocblas_bfloat16>(std::abs(static_cast<float>(a))); }
inline rocblas_bfloat16 sin(const rocblas_bfloat16& a) { return static_cast<rocblas_bfloat16>(std::sin(static_cast<float>(a))); }
inline rocblas_bfloat16 cos(const rocblas_bfloat16& a) { return static_cast<rocblas_bfloat16>(std::cos(static_cast<float>(a))); }
