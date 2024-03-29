//// main.cpp
#include <iostream>
#include <cstdio>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

typedef struct
{
    int c0:8,c1:8,c2:8,c3:8;
} C4;

typedef union
{
    int32_t i;
    C4 c4;
} Int8x4;

__global__ void dot4kernel(const char4 *a, const char4 *b, const int *c, int *d)
{
    int i = hipThreadIdx_x;

    if(i == 0)
    {
        bool saturate = true;
#if (__hcc_workweek__ >= 19015) || __HIP_CLANG_ONLY__
        *d = amd_mixed_dot(*a, *b, *c, saturate);
#endif
    }
}

__global__ void dot4Code(const int *a, const int *b, const int *c, int *d)
{
    int i = hipThreadIdx_x;
    int clamping = 0;

    if(i == 0)
    {
        // below code is alternative to amd_mixed_dot
        d[i] = c[i] 
            + static_cast<int32_t>( a[i] & 0X000000FF      ) * static_cast<int32_t>( b[i] & 0X000000FF      )
            + static_cast<int32_t>((a[i] & 0X0000FF00) >> 8) * static_cast<int32_t>((b[i] & 0X0000FF00) >> 8)
            + static_cast<int32_t>((a[i] & 0X00FF0000) >>16) * static_cast<int32_t>((b[i] & 0X00FF0000) >> 16)
            + static_cast<int32_t>((a[i] & 0XFF000000) >>24) * static_cast<int32_t>((b[i] & 0XFF000000) >> 24);
    }
}

int main()
{
    Int8x4 va, vb, vc, vd;
    int32_t d,c,b[4],a[4];

    vd.i = d = 0x0;
    vc.i = c = 0x00000001;

    vb.c4.c0 = b[0] = 0x1;
    vb.c4.c1 = b[1] = 0x2;
    vb.c4.c2 = b[2] = 0x3;
    vb.c4.c3 = b[3] = 0x4;

    va.c4.c0 = a[0] = 0x5;
    va.c4.c1 = a[1] = 0x6;
    va.c4.c2 = a[2] = 0x7;
    va.c4.c3 = a[3] = 0x8;

    printf("va: %08x\n", va.i);
    printf("vb: %08x\n", vb.i);
    printf("vc: %08x\n", vc.i);
    printf("vd: %08x\n", vd.i);

    // reference calculation of d on host
    d = c + (b[3]*a[3] + b[2]*a[2] + b[1]*a[1] + b[0]*a[0]);

    printf("  host reference result: %08x\n", d);

    int *ga = nullptr, *gb = nullptr, *gc = nullptr, *gd = nullptr;
    char4 *char4_a = nullptr, *char4_b = nullptr;
    hipMalloc(&char4_a, sizeof(va));
    hipMalloc(&char4_b, sizeof(vb));
    hipMalloc(&ga, sizeof(va));
    hipMalloc(&gb, sizeof(vb));
    hipMalloc(&gc, sizeof(vc));
    hipMalloc(&gd, sizeof(vd));

    dim3 grid(1, 1, 1);
    dim3 threads(64, 1, 1);

    // reference calculation of d on device (on dot function)
    hipMemcpy(ga, &va, sizeof(va), hipMemcpyHostToDevice);
    hipMemcpy(gb, &vb, sizeof(vb), hipMemcpyHostToDevice);
    hipMemcpy(gc, &vc, sizeof(vc), hipMemcpyHostToDevice);
    hipMemcpy(gd, &vd, sizeof(vd), hipMemcpyHostToDevice);

    hipLaunchKernelGGL(dot4Code, grid, threads, 0, 0, ga, gb, gc, gd);

    hipMemcpy(&vd, gd, sizeof(vd), hipMemcpyDeviceToHost);

    printf("device reference result: %08x\n", vd.i);

    // calculation of d on device with dot function
    vd.i = d = 0x0;
    hipMemcpy(char4_a, &va, sizeof(va), hipMemcpyHostToDevice);
    hipMemcpy(char4_b, &vb, sizeof(vb), hipMemcpyHostToDevice);
    hipMemcpy(gc, &vc, sizeof(vc), hipMemcpyHostToDevice);
    hipMemcpy(gd, &vd, sizeof(vd), hipMemcpyHostToDevice);

#if (__hcc_workweek__ >= 19015) || __HIP_CLANG_ONLY__
    hipLaunchKernelGGL(dot4kernel, grid, threads, 0, 0, char4_a, char4_b, gc, gd);
#else

#endif

    hipMemcpy(&vd, gd, sizeof(vd), hipMemcpyDeviceToHost);

    printf("device dot4 result: %08x\n", vd.i);

    hipFree(ga);
    hipFree(gb);
    hipFree(gc);
    hipFree(gd);

    return 0;
}
