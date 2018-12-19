//// main.cpp
#include <iostream>
#include <cstdio>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

// extern "C" __device__ int __builtin_amdgcn_sdot4(int, int, int, int );

typedef struct
{
    int c0:8,c1:8,c2:8,c3:8;
} C4;

typedef union
{
    int32_t i;
    C4 c4;
} Int8x4;

__global__ void dot4kernel(const int *a, const int *b, const int *c, int *d)
{
    int i = hipThreadIdx_x;
    int clamping = 0;

    if(i == 0)
    {
//      d[i] = __builtin_amdgcn_sdot4(a[i], b[i], c[i], clamping);
        if (__oclc_ISA_version == 906) 
        {
            d[i] = (float)__llvm_amdgcn_sdot4(a[i], b[i], c[i], true);
        }
        else
        {
            d[i] = c[i] 
                 + static_cast<int32_t>( a[i] & 0X000000FF      ) * static_cast<int32_t>( b[i] & 0X000000FF      )
                 + static_cast<int32_t>((a[i] & 0X0000FF00) >> 8) * static_cast<int32_t>((b[i] & 0X0000FF00) >> 8)
                 + static_cast<int32_t>((a[i] & 0X00FF0000) >>16) * static_cast<int32_t>((b[i] & 0X00FF0000) >> 16)
                 + static_cast<int32_t>((a[i] & 0XFF000000) >>24) * static_cast<int32_t>((b[i] & 0XFF000000) >> 24);
        }
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

    d = c + (b[3]*a[3] + b[2]*a[2] + b[1]*a[1] + b[0]*a[0]);

    printf("  host dot4 result: %08x\n", d);

    int *ga = nullptr, *gb = nullptr, *gc = nullptr, *gd = nullptr;
    hipMalloc(&ga, sizeof(va));
    hipMalloc(&gb, sizeof(vb));
    hipMalloc(&gc, sizeof(vc));
    hipMalloc(&gd, sizeof(vd));

    hipMemcpy(ga, &va, sizeof(va), hipMemcpyHostToDevice);
    hipMemcpy(gb, &vb, sizeof(vb), hipMemcpyHostToDevice);
    hipMemcpy(gc, &vc, sizeof(vc), hipMemcpyHostToDevice);
    hipMemcpy(gd, &vd, sizeof(vd), hipMemcpyHostToDevice);

    dim3 grid(1, 1, 1);
    dim3 threads(64, 1, 1);

    hipLaunchKernelGGL(dot4kernel, grid, threads, 0, 0, ga, gb, gc, gd);

    hipMemcpy(&vd, gd, sizeof(vd), hipMemcpyDeviceToHost);

    printf("device dot4 result: %08x\n", vd.i);

    hipFree(ga);
    hipFree(gb);
    hipFree(gc);
    hipFree(gd);

    return 0;
}
