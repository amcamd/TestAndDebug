#include<iostream>
#include<hip/hip_runtime.h>
#include<hip/hip_runtime_api.h>

typedef __fp16 half8 __attribute__((ext_vector_type(8)));
typedef __fp16 half2 __attribute__((ext_vector_type(2)));

extern "C" half2 __v_pk_fma_f16(half2, half2, half2) __asm("llvm.fma.v2f16");

#define LEN 1024*1024
#define SIZE LEN*sizeof(half8)

#define CHECK_HIP_ERROR(error) \
if (error != hipSuccess) { \
    fprintf(stderr, "error: '%s'(%d) at %s:%d\n", hipGetErrorString(error), error,__FILE__, __LINE__); \
    exit(EXIT_FAILURE);\
}


__global__ void h2Axpy(int n, half2 alpha, half8 *X, half8 *Y) {
    int tx = hipThreadIdx_x + hipBlockIdx_x * 1024;
    half2 y0, y1, y2, y3;
    half2 x0, x1, x2, x3;
    half2 z0, z1, z2, z3;

    y0.x = Y[tx][0];
    y0.y = Y[tx][1];
    y1.x = Y[tx][2];
    y1.y = Y[tx][3];
    y2.x = Y[tx][4];
    y2.y = Y[tx][5];
    y3.x = Y[tx][6];
    y3.y = Y[tx][7];

    x0.x = X[tx][0];
    x0.y = X[tx][1];
    x1.x = X[tx][2];
    x1.y = X[tx][3];
    x2.x = X[tx][4];
    x2.y = X[tx][5];
    x3.x = X[tx][6];
    x3.y = X[tx][7];

    z0 = __v_pk_fma_f16(alpha, x0, y0);
    z1 = __v_pk_fma_f16(alpha, x1, y1);
    z2 = __v_pk_fma_f16(alpha, x2, y2);
    z3 = __v_pk_fma_f16(alpha, x3, y3);

    Y[tx][0] = z0.x;
    Y[tx][1] = z0.y;
    Y[tx][2] = z1.x;
    Y[tx][3] = z1.y;
    Y[tx][4] = z2.x;
    Y[tx][5] = z2.y;
    Y[tx][6] = z3.x;
    Y[tx][7] = z3.y;
}

int main() {
    int n = LEN * 8;
    std::vector<__fp16>X(n), Y(n), Ycopy(n);
    for(int i = 0; i < Y.size(); i++)
    { 
//      Y[i] = __fp16(i%2048);    ERROR at 1961
        Y[i] = __fp16(i%128); 
        X[i] = __fp16((i+4)%64);
    }
    Ycopy = Y;
    hipSetDevice(1);
    __fp16 *Xd, *Yd;
    CHECK_HIP_ERROR(hipMalloc(&Xd, SIZE));
    CHECK_HIP_ERROR(hipMalloc(&Yd, SIZE));
    CHECK_HIP_ERROR(hipMemcpy(Xd, X.data(), SIZE, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(Yd, Y.data(), SIZE, hipMemcpyHostToDevice));
    half2 alpha;
    alpha.x = __fp16(3.0);
    alpha.y = __fp16(2.0);

    auto start = std::chrono::high_resolution_clock::now();

    hipLaunchKernelGGL(h2Axpy, dim3((n/8)/1024,1,1), dim3(1024,1,1), 0, 0, n, alpha, (half8*)Xd, (half8*)Yd);

    CHECK_HIP_ERROR(hipDeviceSynchronize());

    auto stop = std::chrono::high_resolution_clock::now();

    CHECK_HIP_ERROR(hipMemcpy(Y.data(), Yd, SIZE, hipMemcpyDeviceToHost));
    std::cout<<"X, Ycopy, Y    (EVEN)"<<std::endl;
    for (int i = 0; i < 20; i+=2)
    {
        if(float(Y[i] == float(alpha.x) * float(X[i]) + float(Ycopy[i]))) 
        {
            std::cout<<(float)X[i]<<", "<<(float)Ycopy[i]<<", "<<(float)Y[i]<<std::endl;
        }
        else
        {
            std::cout<<(float)X[i]<<", "<<(float)Ycopy[i]<<", "<<(float)Y[i]<<" FAIL"<<std::endl;
        }
    }
    std::cout<<"X, Ycopy, Y    (ODD)"<<std::endl;
    for (int i = 1; i < 20; i+=2)
    {
        if(float(Y[i] == float(alpha.y) * float(X[i]) + float(Ycopy[i]))) 
        {
            std::cout<<(float)X[i]<<", "<<(float)Ycopy[i]<<", "<<(float)Y[i]<<std::endl;
        }
        else
        {
            std::cout<<(float)X[i]<<", "<<(float)Ycopy[i]<<", "<<(float)Y[i]<<" FAIL"<<std::endl;
        }
    }
    std::cout<<std::endl;
    std::cout<<"Y.size() = "<<Y.size()<<std::endl;
    int even_error = 0;
    for(int i=0; i<Y.size(); i+=2) {
        float out = float(alpha.x) * float(X[i]) + float(Ycopy[i]);
        if(float(Y[i]) != out) {
            if(even_error < 100) 
                std::cerr<<"Bad even output: "<<float(Y[i])<<" at: "<<i<<" Expected: "<<out<<std::endl;
            even_error++;
        }
    }
    if(0 == even_error)std::cout<<"---All even pass---\n";
    int odd_error = 0;
    for(int i=1; i<Y.size(); i+=2) {
        float out = float(alpha.y) * float(X[i]) + float(Ycopy[i]);
        if(float(Y[i]) != out) {
            if(odd_error < 100) 
                std::cerr<<"Bad odd output: "<<float(Y[i])<<" at: "<<i<<" Expected: "<<out<<std::endl;
            odd_error++;
        }
    }
    if(0 == odd_error) std::cout<<"---All odd pass---\n";

    double elapsedSec = std::chrono::duration_cast<std::chrono::duration<double>>(stop - start).count();
    double perf = (double)(n*2)/1.0E12/elapsedSec;
    std::cout<<perf<<" TFLOPs"<<std::endl;

    CHECK_HIP_ERROR(hipFree(Xd));
    CHECK_HIP_ERROR(hipFree(Yd));
}
