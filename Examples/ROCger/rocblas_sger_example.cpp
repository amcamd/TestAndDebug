#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <math.h>
#include "rocblas.h"

#define M_DIM 1031
#define N_DIM 1033
#define P_MAX 8

using namespace std;

void printMatrix(const char* name, float* A, rocblas_int m, rocblas_int n, rocblas_int lda) {
    printf("---------- %s ----------\n", name);
    for( int i = 0; i < m; i++) {
        for( int j = 0; j < n; j++) {
            printf("%f ",A[i + j * lda]);
        }
        printf("\n");
    }
}

int main() {

    rocblas_int M = M_DIM;
    rocblas_int N = N_DIM;
    rocblas_int incx = 1;
    rocblas_int incy = 1;
    rocblas_int lda = M_DIM;
    float alpha = 1.0;
    printf("M, N, incx, incy, lda = %d, %d, %d, %d, %d\n",M, N, incx, incy, lda);

    vector<float> hx(M * incx);
    vector<float> hy(N * incy);
    vector<float> hA(N * lda);
    vector<float> hB(N * lda);
    float *dx, *dy, *dA;
    float tolerance = 0, error;

    rocblas_handle handle;
    rocblas_create_handle(&handle);

    // allocate memory on device
    hipMalloc(&dx, M * incx * sizeof(float));
    hipMalloc(&dy, N * incy * sizeof(float));
    hipMalloc(&dA, N * lda  * sizeof(float));

    // Initial Data on CPU,
    srand(1);
    for( int i = 0; i < M; ++i )
    {
        hx[i] = rand() % 10 + 1;  //generate a integer number between [1, 10]
    }
    for( int j = 0; j < N; ++j )
    {
        hy[j * incy] = rand() % 10 + 1;
    }
    for( int i = 0; i < M; ++i )
    {
        for(int j = 0; j < N; ++j)
        {
            hA[i + j * lda] = rand() % 10 + 1;
        }
    }

    // save a copy in hA
    hB = hA;

    hipMemcpy(dx, hx.data(), sizeof(float) * M * incx, hipMemcpyHostToDevice);
    hipMemcpy(dy, hy.data(), sizeof(float) * N * incy, hipMemcpyHostToDevice);
    hipMemcpy(dA, hA.data(), sizeof(float) * N * lda,  hipMemcpyHostToDevice);

    rocblas_sger(handle, M, N, &alpha, dx, incx, dy, incy, dA, lda);

    // copy output from device memory to host memory
    hipMemcpy(hA.data(), dA, sizeof(float) * N * lda, hipMemcpyDeviceToHost);

    // verify rocblas_scal result
    for(rocblas_int i=0; i<M; i++)
    {
        for(rocblas_int j=0; j<N; j++)
        {
            hB[i+j*lda] += alpha * hx[i*incx] * hy[j*incy];
            error = fabs(hB[i+j*lda] - hA[i+j*lda]);
            if(error > tolerance)
            {
              printf("error %d,%d: %f  CPU=%f, GPU=%f\n", i,j,error, hB[i+j*lda], hA[i+j*lda]);
//            break;
            }
        }
    }
    printMatrix("calculated matrix ha", hA.data(), min(P_MAX,M), min(P_MAX,N), lda);
    printMatrix("reference  matrix hb", hB.data(), min(P_MAX,M), min(P_MAX,N), lda);

    hipFree(dx);
    hipFree(dy);
    hipFree(dA);
    rocblas_destroy_handle(handle);

    if(error > tolerance){
        printf("GER Failed !\n");
        return 1;
    }
    else{
        printf("GER Success !\n");
        return 0;
    }

}
