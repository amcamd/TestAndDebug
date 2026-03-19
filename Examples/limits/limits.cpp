#include <limits>
#include <iostream>
#include <cstdint>

int main() {

    float max_value = std::numeric_limits<float>::max();

    float inf_value = max_value + max_value;
    float Ninf_value = -inf_value;
    float NaN_value = inf_value + Ninf_value;
    float not_NaN_value = max_value + 1.0f;

    std::cout << "    inf value = " << inf_value << std::endl;
    std::cout << "   -inf value = " <<    Ninf_value << std::endl;
    std::cout << "    NaN value = " <<     NaN_value << std::endl;
    std::cout << "not NaN value = " << not_NaN_value << std::endl;

    float accumulator = 0.0f;
    float accumulator_previous = accumulator;
    uint64_t uint64_t_max = std::numeric_limits<uint64_t>::max();
    for(uint64_t i = 0; i < uint64_t_max; i++)
    {
	accumulator = accumulator + 1.0f;
	if(accumulator == accumulator_previous)
	{
	    break;
	}
	else
	{
	    accumulator_previous = accumulator;
	}
    }
    std::cout << std::endl;
    std::cout << "float(uint64_t_max) = " << static_cast<float>(uint64_t_max) << std::endl;
    std::cout << "accumulator = " << accumulator << std::endl;


    float eps = std::numeric_limits<float>::epsilon();
    std::cout << "float eps = " << eps << std::endl;

    float accumulator_min2max = 0.0f;
    float accumulator_max2min = 0.0f;
    uint64_t i_limit_small = 1000000;
    uint64_t i_limit_large = 10000000;
//  uint64_t i_limit_small = 100;
//  uint64_t i_limit_large = 1000;
    uint64_t arithmetic_series_sum = ((i_limit_small + i_limit_large) * (i_limit_large - i_limit_small + 1)) / 2;
    for(int i = i_limit_small; i <= i_limit_large; i++)
    {
        accumulator_min2max += static_cast<float>(i);
    }
    for(int i = i_limit_large; i >= i_limit_small; i--)
    {
        accumulator_max2min += static_cast<float>(i);
    }
    std::cout << i_limit_small << " + ... + " << i_limit_large << " = " << arithmetic_series_sum << std::endl;
    std::cout << "accumulator_min2max = " << accumulator_min2max << std::endl;
    std::cout << "accumulator_max2min = " << accumulator_max2min << std::endl;
    
    return 0;
}
