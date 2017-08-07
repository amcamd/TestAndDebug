#include <hip/hip_runtime.h>
#include <stdio.h>
#define NB 256

typedef __fp16 f_type;
// typedef float f_type;

#define CHECK_HIP_ERROR(error) \
    if (error != hipSuccess) { \
        fprintf(stderr, "error: '%s'(%d) at %s:%d\n", hipGetErrorString(error), error,__FILE__, __LINE__); \
        exit(EXIT_FAILURE);\
    }

__global__
void saxpy(hipLaunchParm lp, int n, f_type alpha, f_type *x, f_type *y)
{
    int tid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if (tid < n) y[tid] = alpha * x[tid] + y[tid];
}

int main(void)
{
    int N = 500;
    f_type alpha = 2.0;
    f_type *x, *y, *d_x, *d_y;
    x = (f_type*)malloc(N*sizeof(f_type));
    y = (f_type*)malloc(N*sizeof(f_type));
  
    CHECK_HIP_ERROR(hipMalloc(&d_x, N*sizeof(f_type))); 
    CHECK_HIP_ERROR(hipMalloc(&d_y, N*sizeof(f_type)));
  
    for (int i = 0; i < N; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }
  
    CHECK_HIP_ERROR(hipMemcpy(d_x, x, N*sizeof(f_type), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_y, y, N*sizeof(f_type), hipMemcpyHostToDevice));
  
    // Perform SAXPY on 1M elements
    int blocks = (N - 1) / NB + 1;
    dim3 grid(blocks, 1, 1);
    dim3 threads(NB, 1, 1);
    hipLaunchKernel(saxpy, dim3(grid), dim3(threads), 0, 0, N, 2.0f, d_x, d_y);
  
    CHECK_HIP_ERROR(hipMemcpy(y, d_y, N*sizeof(f_type), hipMemcpyDeviceToHost));
  
    f_type maxError = 0.0f;
    for (int i = 0; i < min(10, N); i++)
        printf("i, y[i] = %d, %f\n", i, y[i]);
    for (int i = 0; i < N; i++)
      maxError = max(maxError, std::abs(y[i]-4.0f));
    printf("Max error: %f\n", maxError);
  
    CHECK_HIP_ERROR(hipFree(d_x));
    CHECK_HIP_ERROR(hipFree(d_y));
    free(x);
    free(y);
}
