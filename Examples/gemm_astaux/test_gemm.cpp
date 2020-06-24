
#include <complex>
#include <iostream>
#include <string>

#include <hip/hip_runtime.h>
#include <omp.h>

#include "astaux.h"

#define HIP_CHECK(status)                                                                \
    if (status != hipSuccess) {                                                          \
        std::cout << "Got Status: " << status << " at Line: " << __LINE__ << std::endl;  \
        exit(0);                                                                         \
    }


//----------------------------------------------------------------------------
template <typename T>
void test_gemm(int m,int n, int k, int lda, int ldb, int ldc,
               int batch_count, int iterations, int pattern)
{
    //----------
    // GPU setup
    std::size_t size_A = lda*k*batch_count;
    std::size_t size_B = ldb*n*batch_count;
    std::size_t size_C = ldc*n*batch_count;
    T* h_A = (T*)malloc(sizeof(T)*size_A);
    T* h_B = (T*)malloc(sizeof(T)*size_B);
    T* h_C = (T*)malloc(sizeof(T)*size_C);
    assert(h_A != nullptr);
    assert(h_B != nullptr);
    assert(h_C != nullptr);


    T* d_A;
    T* d_B;
    T* d_C;
    HIP_CHECK(hipMalloc(&d_A, sizeof(T)*size_A));
    HIP_CHECK(hipMalloc(&d_B, sizeof(T)*size_B));
    HIP_CHECK(hipMalloc(&d_C, sizeof(T)*size_C));
    T** h_Aptr = (T**)malloc(sizeof(T*)*batch_count);
    T** h_Bptr = (T**)malloc(sizeof(T*)*batch_count);
    T** h_Cptr = (T**)malloc(sizeof(T*)*batch_count);
    assert(h_Aptr != nullptr);
    assert(h_Bptr != nullptr);
    assert(h_Cptr != nullptr);
    for (int i = 0; i < batch_count; ++i) {
        h_Aptr[i] = d_A + lda*k*i;
        h_Bptr[i] = d_B + ldb*n*i;
        h_Cptr[i] = d_C + ldc*n*i;
    }

    T** d_Aptr;
    T** d_Bptr;
    T** d_Cptr;
    HIP_CHECK(hipMalloc(&d_Aptr, sizeof(T*)*batch_count));
    HIP_CHECK(hipMalloc(&d_Bptr, sizeof(T*)*batch_count));
    HIP_CHECK(hipMalloc(&d_Cptr, sizeof(T*)*batch_count));
    HIP_CHECK(hipMemcpy(d_Aptr, h_Aptr, sizeof(T*)*batch_count,
                       hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_Bptr, h_Bptr, sizeof(T*)*batch_count,
                       hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_Cptr, h_Cptr, sizeof(T*)*batch_count,
                       hipMemcpyHostToDevice));

    int iseed[4] = {0, 0, 0, 1};
//  LAPACKE_larnv(2, iseed, size_A, h_A);
//  LAPACKE_larnv(2, iseed, size_B, h_B);
//  LAPACKE_larnv(2, iseed, size_C, h_C);

//  hipblasHandle_t handle;
//  hipblasCreate(&handle);

    T alpha;
    T beta;
    switch (pattern) {
        case 1:
            alpha = T(1.0);
            beta = T(0.0);
            break;
        case 2:
            alpha = T(1.0);
            beta = T(1.0);
            break;
        case 3:
            alpha = T(1.0);
            beta = T(1.0);
            break;
        case 4:
            alpha = T(-1.0);
            beta = T(0.0);
            break;
        case 5:
            alpha = T(1.0);
            beta = T(0.0);
            break;
        default: assert(false);
    }

    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));

    //--------
    // GPU run
    double min_time = std::numeric_limits<double>::infinity();

    T* Cref = (T*)malloc(sizeof(T)*size_C);

    for (int i = 0; i < iterations; ++i) {

        HIP_CHECK(hipMemcpy(d_A, h_A, sizeof(T)*size_A, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_B, h_B, sizeof(T)*size_B, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_C, h_C, sizeof(T)*size_C, hipMemcpyHostToDevice));

        double start;
        double elapsed;
        start = omp_get_wtime();

            astGemmBatched<T>(m, n, k,
                              &alpha, d_Aptr, lda,
                                      d_Bptr, ldb,
                              &beta,  d_Cptr, ldc,
                              batch_count, pattern,
                              stream);
            hipDeviceSynchronize();
        elapsed = omp_get_wtime() - start;
        if (elapsed < min_time)
            min_time = elapsed;
    }


    double ops;
    double bytes;
    ops = 2.0*m*n*k*batch_count;
    if (std::is_same<T, std::complex<float>>::value ||
        std::is_same<T, std::complex<double>>::value  )
    {
        ops *= 4;
    }
    printf("\t%lf\n", ops/min_time/1e9);

    bytes = sizeof(T)*m*n + sizeof(T)*m*k + sizeof(T)*k*n;
    if (beta != T(0.0))
        bytes += sizeof(T)*m*n;
    bytes *= batch_count;
    printf("\t%lf\n", bytes/min_time/1e9);

    //------------------
    // CPU setup and run
    assert(Cref != nullptr);
    memcpy(Cref, h_C, sizeof(T)*size_C);

    // compare GPU (h_C) to CPU (Cref)
    HIP_CHECK(hipMemcpy(h_C, d_C, sizeof(T)*size_C, hipMemcpyDeviceToHost));
    // batch_print(m, n, Cref, ldc, batch_count);
    // batch_print(m, n, h_C, ldc, batch_count);
//  batch_diff(m, n, Cref, ldc, h_C, ldc, batch_count);


    //--------
    // cleanup
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_Aptr);
    free(h_Bptr);
    free(h_Cptr);
    free(Cref);

    hipFree(d_A);
    hipFree(d_B);
    hipFree(d_C);
    hipFree(d_Aptr);
    hipFree(d_Bptr);
    hipFree(d_Cptr);

    HIP_CHECK(hipStreamDestroy(stream));
}


//----------------------------------------------------------------------------
int main(int argc, char** argv)
{
//  assert(argc == 10);
//  int m = std::atoi(argv[1]);
//  int n = std::atoi(argv[2]);
//  int k = std::atoi(argv[3]);
//  int lda = std::atoi(argv[4]);
//  int ldb = std::atoi(argv[5]);
//  int ldc = std::atoi(argv[6]);
//  int batch_count = std::atoi(argv[7]);
//  int iterations = std::atoi(argv[8]);
//  int pattern = std::atoi(argv[9]);
//  assert(argc == 10);
    int m = 128;
    int n = 128;
    int k = 128;
    int lda = 128;
    int ldb = 128;
    int ldc = 128;
    int batch_count = 2;
    int iterations = 2;
    int pattern = 1;
    assert(lda >= m);
    assert(ldb >= k);
    assert(ldc >= m);

    test_gemm<float>(
        m, n, k, lda, ldb, ldc, batch_count, iterations, pattern);
    printf("\n");

    return (EXIT_SUCCESS);
}
