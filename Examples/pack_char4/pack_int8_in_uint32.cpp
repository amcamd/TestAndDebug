#include<iostream>
#include <cinttypes>

inline int pack4chars(char c_0, char c_1, char c_2, char c_3)
{
    return ((int)(((unsigned char)c_0) << 24)
    |  (int)(((unsigned char)c_1) << 16)
    |  (int)(((unsigned char)c_2) << 8)
    |  (int)((unsigned char)c_3));
}

typedef union{
    int8_t byte[4];
    uint32_t uval;
    int32_t val;
} int8x4;

void unpack_int8x4(uint32_t in, int32_t &out_0, int32_t &out_1, int32_t &out_2, int32_t &out_3)
{
    int8x4 x;
    x.uval = in;
    out_0 = x.byte[0];
    out_1 = x.byte[1];
    out_2 = x.byte[2];
    out_3 = x.byte[3];
}

int main(int argc, char *argv[]) 
{
    char c_0 = 1;
    char c_1 = -2;
    char c_2 = 3;
    char c_3 = -4;

//  uint32_t input = 0x7F8065A3;
    uint32_t input = pack4chars(c_0, c_1, c_2, c_3);
    int32_t out_0=0, out_1=0, out_2=0, out_3=0;
    unpack_int8x4(input, out_0, out_1, out_2, out_3);
    std::cout << "out_0,1,2,3 = " << out_0 << ", " << out_1 << ", " << out_2 << ", " << out_3 << std::endl;

    uint32_t four_32bit_1 = pack4chars(c_0, c_1, c_2, c_3);

    std::cout << "four_32bit_1 hex = " << std::hex << four_32bit_1 << std::endl;

//  int32_t val;
//  val = 0x7F8065A3;

//  xv = static_cast<signed char *>(&val)[0]; std::cout << " : xv_0 = " << xv;
//  xv = static_cast<signed char *>(&val)[1]; std::cout << " : xv_1 = " << xv;
//  xv = static_cast<signed char *>(&val)[2]; std::cout << " : xv_2 = " << xv;
//  xv = static_cast<signed char *>(&val)[3]; std::cout << " : xv_3 = " << xv;
//  std::cout << std::endl;





    char     *A_quad = reinterpret_cast<char    *>(&four_32bit_1);

    // extract 4 int32_t from 4 int8 inside uint32_t
    int32_t A_0 = static_cast<char>(A_quad[0]) & 0x000000FF;
    int32_t A_1 = static_cast<char>(A_quad[1]) & 0x000000FF;
    int32_t A_2 = static_cast<char>(A_quad[2]) & 0x000000FF;
    int32_t A_3 = static_cast<char>(A_quad[3]) & 0x000000FF;

    std::cout << std::dec;
    std::cout << "    A_0, A_1, A_2, A_3 = " << A_0 << ", " << A_1 << ", " << A_2 << ", " << A_3 << std::endl;

    // sign extend int8 values in int32 datatype
    if(A_0 & 0x00000080) A_0 = A_0 | 0xFFFFFF00;
    if(A_1 & 0x00000080) A_1 = A_1 | 0xFFFFFF00;
    if(A_2 & 0x00000080) A_2 = A_2 | 0xFFFFFF00;
    if(A_3 & 0x00000080) A_3 = A_3 | 0xFFFFFF00;

    std::cout << "    A_0, A_1, A_2, A_3 = " << A_0 << ", " << A_1 << ", " << A_2 << ", " << A_3 << std::endl;

    c_0 = -4;
    c_1 = -1;
    c_2 = -2;
    c_3 = -3;

    four_32bit_1 = pack4chars(c_0, c_1, c_2, c_3);

    std::cout << "four_32bit_1 hex = " << std::hex << four_32bit_1 << std::endl;

    char     *B_quad = reinterpret_cast<char    *>(&four_32bit_1);

    // extract 4 int32_t from 4 int8 inside uint32_t
    int32_t B_0 = static_cast<char>(B_quad[0]) & 0x000000FF;
    int32_t B_1 = static_cast<char>(B_quad[1]) & 0x000000FF;
    int32_t B_2 = static_cast<char>(B_quad[2]) & 0x000000FF;
    int32_t B_3 = static_cast<char>(B_quad[3]) & 0x000000FF;

    std::cout << std::dec;
    std::cout << "    B_0, B_1, B_2, B_3 = " << B_0 << ", " << B_1 << ", " << B_2 << ", " << B_3 << std::endl;
    // sign extend int8 values in int32 datatype
    if(B_0 & 0x00000080) B_0 = B_0 | 0xFFFFFF00;
    if(B_1 & 0x00000080) B_1 = B_1 | 0xFFFFFF00;
    if(B_2 & 0x00000080) B_2 = B_2 | 0xFFFFFF00;
    if(B_3 & 0x00000080) B_3 = B_3 | 0xFFFFFF00;

    std::cout << "    B_0, B_1, B_2, B_3 = " << B_0 << ", " << B_1 << ", " << B_2 << ", " << B_3 << std::endl;

    return 0;
}
