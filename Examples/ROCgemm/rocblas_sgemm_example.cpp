#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <math.h>
#include <chrono>
#include <limits>
#include "rocblas.h"

#define M_DIM 1000
#define N_DIM 128
#define K_DIM 4096
#define TRANS_A rocblas_operation_transpose
#define TRANS_B rocblas_operation_none
#define ALPHA 1.0
#define BETA 1.0
#define P_MAX 8
#define PERFORMANCE_TEST true
#define CORRECTNESS_TEST true

#define CHECK_HIP_ERROR(error) \
    if (error != hipSuccess) { \
      fprintf(stderr, "error: '%s'(%d) at %s:%d\n", hipGetErrorString(error), error,__FILE__, __LINE__); \
      exit(EXIT_FAILURE);\
    }

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

template <typename T>
void Mat_mat_mult(T alpha, T beta, int M, int N, int K, vector<T> A, int As1, int As2, 
                  vector<T> B, int Bs1, int Bs2, vector<T> & C, int Cs1, int Cs2) {
    for(int i1=0; i1<M; i1++) {
        for(int i2=0; i2<N; i2++) {
            float t = 0.0;
            for(int i3=0; i3<K; i3++){
                t +=  A[i1 * As1 + i3 * As2] * B[i3 * Bs1 + i2 * Bs2]; 
            }
            C[i1*Cs1 +i2*Cs2] = alpha * C[i1*Cs1+i2*Cs2] + beta * t ;
        }
    }
}

int main() {

    rocblas_int M = M_DIM, N = N_DIM, K = K_DIM;
    rocblas_operation transA = TRANS_A, transB = TRANS_B;
    float alpha = ALPHA, beta  = BETA;

    rocblas_status status = rocblas_status_success;
    rocblas_order order = rocblas_order_column_major;
    std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
    std::chrono::duration<double> dur;

    rocblas_int lda, ldb, ldc, sizeOfA, sizeOfB, sizeOfC, As1, As2, Bs1, Bs2;
    ldc = M;
    sizeOfC = N * ldc;
    if( transA == rocblas_operation_none){
        lda = M; As1 = 1; As2 = lda; sizeOfA = K * lda; printf("N");
    }
    else {
        lda = K; As1 = lda; As2 = 1; sizeOfA = M * lda; printf("T");
    }
    if( transB == rocblas_operation_none){
        ldb = K; Bs1 = 1; Bs2 = ldb; sizeOfB = N * ldb; printf("N:");
    }
    else {
        ldb = N; Bs1 = ldb; Bs2 = 1; sizeOfB = K * ldb; printf("T: ");
    }
 
    printf("M, N, K, lda, ldb, ldc = %d, %d, %d, %d, %d, %d\n",M, N, K, lda, ldb, ldc);

    vector<float> hA(sizeOfA), hB(sizeOfB), hC(sizeOfC), hC_copy(sizeOfC);
    float *dA, *dB, *dC;
    float tolerance = 0, error;

    rocblas_handle handle;
    rocblas_create_handle(&handle);

    // allocate memory on device
    CHECK_HIP_ERROR(hipMalloc(&dA, sizeOfA * sizeof(float)));
    CHECK_HIP_ERROR(hipMalloc(&dB, sizeOfB * sizeof(float)));
    CHECK_HIP_ERROR(hipMalloc(&dC, sizeOfC * sizeof(float)));

    // Initial Data on CPU,
    srand(1);
    for( int i = 0; i < sizeOfA; ++i ) { hA[i] = rand() % 10 + 1; }
    for( int i = 0; i < sizeOfB; ++i ) { hB[i] = rand() % 10 + 1; }
    for( int i = 0; i < sizeOfC; ++i ) { hC[i] = rand() % 10 + 1; }

    // save a copy in hA
    hC_copy = hC;

    CHECK_HIP_ERROR(hipMemcpy(dA, hA.data(), sizeof(float) * sizeOfA, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dB, hB.data(), sizeof(float) * sizeOfB, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dC, hC.data(), sizeof(float) * sizeOfC, hipMemcpyHostToDevice));

    if(CORRECTNESS_TEST){
        status = rocblas_sgemm(handle, order, transA, transB, M, N, K, &alpha, dA, lda, dB, 
                               ldb, &beta, dC, ldc);

        if( status != rocblas_status_success) {
            printf("***ERROR***: rocblas_status = %d\n", status);
            return -1;
        }

        // copy output from device memory to host memory
        CHECK_HIP_ERROR(hipMemcpy(hC.data(), dC, sizeof(float) * sizeOfC, hipMemcpyDeviceToHost));

        Mat_mat_mult<float>(alpha, beta, M, N, K, hA, As1, As2, hB, Bs1, Bs2, hC_copy, 1, ldc);

//      printMatrix("calculated matrix hC", hC.data(), min(P_MAX,M), min(P_MAX,N), ldc);
//      printMatrix("reference  matrix hC_copy", hC_copy.data(), min(P_MAX,M), min(P_MAX,N), ldc);

        // verify rocblas_scal result
        for(rocblas_int i1=0; i1<M; i1++) {
            for(rocblas_int i2=0; i2<N; i2++) {
                error = fabs(hC[i1+i2*ldc] - hC_copy[i1+i2*ldc]);
                if(error > tolerance) {
                  printf("error %d,%d: %f  CPU=%f, GPU=%f\n",i1,i2,error,hC[i1+i2*lda],hC_copy[i1+i2*lda]);
                  break;
                }
            }
            if(error > tolerance) break;
        }

        if(error > tolerance){
            printf("GEMM Failed !\n"); return 1;
        }
        else{
            printf("GEMM Success !\n");
        }
    }

    if(PERFORMANCE_TEST) {

        start = std::chrono::high_resolution_clock::now();
        rocblas_sgemm(handle, order, transA, transB, M, N, K, &alpha, dA, lda, dB, ldb, &beta, dC, ldc);
        end = std::chrono::high_resolution_clock::now();
        dur= end - start;

        int number_iterations = 10.0 > (1.0 / dur.count()) ? 10 : (int) (1.0 / dur.count());

        double min_seconds = numeric_limits<double>::max();
        for(int i = 0; i < number_iterations; i++){
            start = std::chrono::high_resolution_clock::now();

            rocblas_sgemm(handle, order, transA, transB, M, N, K, &alpha, dA, lda, dB, ldb, &beta, dC, ldc);

            end = std::chrono::high_resolution_clock::now();
            dur= end - start;
            min_seconds = min_seconds < dur.count() ? min_seconds : dur.count();
        }
        double gflops = (double)(M) * (double)(N) * (double)(K) * 2.0 / min_seconds / 1e9;
        printf("seconds, gflops, number_iterations = %f, %f, %d\n", min_seconds, gflops, number_iterations);
    }

    CHECK_HIP_ERROR(hipFree(dA));
    CHECK_HIP_ERROR(hipFree(dB));
    CHECK_HIP_ERROR(hipFree(dC));
    rocblas_destroy_handle(handle);

    return 0;
}
