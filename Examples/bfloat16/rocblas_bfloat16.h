#include <cmath>
#include <math.h>

#ifndef __BYTE_ORDER__
#define __BYTE_ORDER__ __ORDER_LITTLE_ENDIAN__
#endif

typedef struct rocblas_bfloat16
{
    rocblas_bfloat16() : data(0){}

    // truncate lower 16 bits of IEEE float to convert to bfloat16
    static rocblas_bfloat16 float_to_bfloat16(const float v) 
    {
        rocblas_bfloat16 bf16;
        if (std::isnan(v))
        {
            bf16.data = Q_NAN_VALUE;
            return bf16;
        }
        const uint16_t* p_ui16= reinterpret_cast<const uint16_t*>(&v);
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
        bf16.data = p_ui16[0];
#else
        bf16.data = p_ui16[1];
#endif
        return bf16;
    }

    explicit rocblas_bfloat16(const float v) 
    {
        data = float_to_bfloat16(v).data;
    }

    // zero extend lower 16 bits of bfloat16 to convert to IEEE float
    explicit operator float() const
    {
        float fp32 = 0;

        uint16_t* q = reinterpret_cast<uint16_t*>(&fp32);

#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
        q[0] = data;
#else
        q[1] = data;
#endif
        return fp32;
    }

    bool is_zero() const 
    { 
        return (data & 0x7FFF) == ZERO_VALUE; 
    }

    uint16_t data;

    private:

    static const uint16_t Q_NAN_VALUE = 0xFFC1;

    static const uint16_t ZERO_VALUE = 0;

} rocblas_bfloat16;

inline std::ostream& operator<<(std::ostream& os, const rocblas_bfloat16& bf16) { os << static_cast<float>(bf16); return os; }

inline rocblas_bfloat16 operator+(rocblas_bfloat16 a, rocblas_bfloat16 b) { return rocblas_bfloat16(static_cast<float>(a) + static_cast<float>(b)); }
inline rocblas_bfloat16 operator-(rocblas_bfloat16 a, rocblas_bfloat16 b) { return rocblas_bfloat16(static_cast<float>(a) - static_cast<float>(b)); }
inline rocblas_bfloat16 operator*(rocblas_bfloat16 a, rocblas_bfloat16 b) { return rocblas_bfloat16(static_cast<float>(a) * static_cast<float>(b)); }
inline rocblas_bfloat16 operator/(rocblas_bfloat16 a, rocblas_bfloat16 b) { return rocblas_bfloat16(static_cast<float>(a) / static_cast<float>(b)); }
inline bool operator<(rocblas_bfloat16 a, rocblas_bfloat16 b) { return static_cast<float>(a) < static_cast<float>(b); }
inline bool operator<=(rocblas_bfloat16 a, rocblas_bfloat16 b) { return static_cast<float>(a) <= static_cast<float>(b); }
inline bool operator==(rocblas_bfloat16 a, rocblas_bfloat16 b) { return static_cast<float>(a) == static_cast<float>(b); }
inline bool operator!=(rocblas_bfloat16 a, rocblas_bfloat16 b) { return static_cast<float>(a) != static_cast<float>(b); }
inline bool operator>(rocblas_bfloat16 a, rocblas_bfloat16 b) { return static_cast<float>(a) > static_cast<float>(b); }
inline bool operator>=(rocblas_bfloat16 a, rocblas_bfloat16 b) { return static_cast<float>(a) >= static_cast<float>(b); }
inline rocblas_bfloat16& operator+=(rocblas_bfloat16& a, rocblas_bfloat16 b) { a = a + b; return a; }
inline rocblas_bfloat16& operator-=(rocblas_bfloat16& a, rocblas_bfloat16 b) { a = a - b; return a; }
inline rocblas_bfloat16 operator++(rocblas_bfloat16& a) { a += rocblas_bfloat16(1); return a; }
inline rocblas_bfloat16 operator--(rocblas_bfloat16& a) { a -= rocblas_bfloat16(1); return a; }
inline rocblas_bfloat16 operator++(rocblas_bfloat16& a, int) { rocblas_bfloat16 original_value = a; ++a; return original_value; }
inline rocblas_bfloat16 operator--(rocblas_bfloat16& a, int) { rocblas_bfloat16 original_value = a; --a; return original_value; }
inline rocblas_bfloat16& operator*=(rocblas_bfloat16& a, rocblas_bfloat16 b) { a = a * b; return a; }
inline rocblas_bfloat16& operator/=(rocblas_bfloat16& a, rocblas_bfloat16 b) { a = a / b; return a; }

inline bool isinf(const rocblas_bfloat16& a) { return std::isinf(float(a)); }
inline bool isnan(const rocblas_bfloat16& a) { return std::isnan(float(a)); }
inline rocblas_bfloat16 abs(const rocblas_bfloat16& a) { return rocblas_bfloat16(std::abs(float(a))); }
inline rocblas_bfloat16 sin(const rocblas_bfloat16& a) { return rocblas_bfloat16(std::sin(float(a))); }
inline rocblas_bfloat16 cos(const rocblas_bfloat16& a) { return rocblas_bfloat16(std::cos(float(a))); }
