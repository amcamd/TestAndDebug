#include <iostream>
#include <iomanip>


int main(int argc, char* argv[])
{
    float a = static_cast<float>(1 << 24) - 10;

    for (int i = 0; i < 40; i++)
    {
        std::cout << std::setprecision(10) << "a, i = " << ++a << "  " << i << std::endl;
    }
    return 0;
}
