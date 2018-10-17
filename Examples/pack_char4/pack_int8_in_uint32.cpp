#include<iostream>

inline int pack4chars(char c1, char c2, char c3, char c4)
{
    return ((int)(((unsigned char)c1) << 24)
    |  (int)(((unsigned char)c2) << 16)
    |  (int)(((unsigned char)c3) << 8)
    |  (int)((unsigned char)c4));
}

int main(int argc, char *argv[]) 
{
    char c1 = 1;
    char c2 = 2;
    char c3 = 3;
    char c4 = 4;

    uint32_t four_32bit_1 = pack4chars(c1, c2, c3, c4);

    std::cout << "four_32bit_1 hex = " << std::hex << four_32bit_1 << std::endl;

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

    c1 = -1;
    c2 = -2;
    c3 = -3;
    c4 = -4;

    four_32bit_1 = pack4chars(c1, c2, c3, c4);

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
