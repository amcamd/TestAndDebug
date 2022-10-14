
void my_function_v2(float a, float b);

inline void my_function(float a, float b)
{
    my_function_v2(a, b);
}
