#include <stdio.h>
#include <atomic>
#include <hip/hip_runtime.h>

#define CHECK(cmd) \
{\
    hipError_t error  = cmd;\
    if (error != hipSuccess) { \
      fprintf(stderr, "error: '%s'(%d) at %s:%d\n", hipGetErrorString(error), error,__FILE__, __LINE__); \
    exit(EXIT_FAILURE);\
    }\
}

template <typename T, typename std::enable_if<std::is_arithmetic<T>::value>::type* = nullptr>
__device__ inline void atomic_add(T *fPtr, T operand)
{
    std::atomic<T> *aPtr = reinterpret_cast<std::atomic<T>*>(fPtr);
    T oldValue, newValue;
    oldValue = aPtr->load(std::memory_order_relaxed);
    do
    {
      newValue = oldValue + operand;
    }
    while ( !std::atomic_compare_exchange_weak_explicit(
                aPtr, 
                &oldValue, 
                newValue, 
                std::memory_order_relaxed, 
                std::memory_order_release) );
}

template <typename T>
__global__ void
test_atomic_add(hipLaunchParm lp, T *accumulator, T increment)
{
    atomic_add(accumulator, increment);
}

int main()
{
    const unsigned blocks = 256;
    const unsigned threadsPerBlock = 256;
    int increment = 1;

    dim3 grid(blocks,1,1);
    dim3 threads(threadsPerBlock,1,1);
    hipStream_t kernel_stream;

    int    h_acc_i32 = 0, h_inc_i32 = static_cast<int>(increment);
    float  h_acc_f32 = 0, h_inc_f32 = static_cast<float>(increment);
    double h_acc_f64 = 0, h_inc_f64 = static_cast<double>(increment);
    int    *d_acc_i32;
    float  *d_acc_f32;
    double *d_acc_f64;

    CHECK(hipMalloc(&d_acc_i32, sizeof(int)));
    CHECK(hipMalloc(&d_acc_f32, sizeof(float)));
    CHECK(hipMalloc(&d_acc_f64, sizeof(double)));

    CHECK(hipMemcpy(d_acc_i32, &h_acc_i32, sizeof(int), hipMemcpyHostToDevice));
    CHECK(hipMemcpy(d_acc_f32, &h_acc_f32, sizeof(float), hipMemcpyHostToDevice));
    CHECK(hipMemcpy(d_acc_f64, &h_acc_f64 , sizeof(double), hipMemcpyHostToDevice));

    //kernelCall
    hipLaunchKernel(test_atomic_add, dim3(grid), dim3(threads), 0, 0,
                d_acc_i32, h_inc_i32);

    //kernelCall
    hipLaunchKernel(test_atomic_add, dim3(grid), dim3(threads), 0, 0,
                d_acc_f32, h_inc_f32);

    //kernelCall
    hipLaunchKernel(test_atomic_add, dim3(grid), dim3(threads), 0, 0,
                d_acc_f64, h_inc_f64);

    CHECK(hipMemcpy(&h_acc_i32, d_acc_i32, sizeof(int), hipMemcpyDeviceToHost));
    CHECK(hipMemcpy(&h_acc_f32, d_acc_f32, sizeof(float), hipMemcpyDeviceToHost));
    CHECK(hipMemcpy(&h_acc_f64, d_acc_f64, sizeof(double), hipMemcpyDeviceToHost));

    CHECK(hipFree(d_acc_i32));
    CHECK(hipFree(d_acc_f32));
    CHECK(hipFree(d_acc_f64));

    std::cout << "expected result      = " << increment << "  " << increment * blocks * threadsPerBlock << std::endl;
    std::cout << "h_inc_i32, h_acc_i32 = " << h_inc_i32 << "  " << h_acc_i32 << std::endl;
    std::cout << "h_inc_f32, h_acc_f32 = " << h_inc_f32 << "  " << h_acc_f32 << std::endl;
    std::cout << "h_inc_f64, h_acc_f64 = " << h_inc_f64 << "  " << h_acc_f64 << std::endl;
}
