#include <stdlib.h>
#include <stdio.h>
#include <atomic>
#include <hip/hip_runtime.h>

template <typename T>
__device__ inline void atomicAddFloat(T *fPtr, T operand)
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
atomicAddFloat(hipLaunchParm lp, T *accumulator, T increment)
{
    atomicAddFloat(accumulator, increment);
}

int main()
{
    const unsigned blocks = 4;
    const unsigned threadsPerBlock = 256;

    dim3 grid(blocks,1,1);
    dim3 threads(threadsPerBlock,1,1);
    hipStream_t kernel_stream;

    float  h_acc_32 = 0, h_inc_32 = 1;
    double h_acc_64 = 0, h_inc_64 = 1;
    float  *d_acc_32;
    double *d_acc_64;

    hipMalloc(&d_acc_32, sizeof(float));
    hipMalloc(&d_acc_64, sizeof(double));

    hipMemcpy(d_acc_32, &h_acc_32, sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_acc_64, &h_acc_64 , sizeof(double), hipMemcpyHostToDevice);

    //kernelCall
    hipLaunchKernel(atomicAddFloat, dim3(grid), dim3(threads), 0, 0, 
                d_acc_32, h_inc_32);

    //kernelCall
    hipLaunchKernel(atomicAddFloat, dim3(grid), dim3(threads), 0, 0, 
                d_acc_64, h_inc_64);

    hipMemcpy(&h_acc_32, d_acc_32, sizeof(float), hipMemcpyDeviceToHost);
    hipMemcpy(&h_acc_64, d_acc_64, sizeof(double), hipMemcpyDeviceToHost);

    std::cout << "h_inc_32, h_acc_32 = " << h_inc_32 << "  " << h_acc_32 << std::endl;
    std::cout << "h_inc_64, h_acc_64 = " << h_inc_64 << "  " << h_acc_64 << std::endl;
}
