/// \file
/// \brief batched routines for tiny matrices
/// \author Jakub Kurzak

#include <complex>
#include <hip/hip_runtime.h>
#include <iostream>
#include "rocblas.h"

// general alpha, beta, m, n, k
template <typename T,
          int DIM_M, int DIM_N,
          int BLK_M, int BLK_N, int BLK_K,
          int DIM_M_A, int DIM_N_A,
          int DIM_M_B, int DIM_N_B>
__attribute__((amdgpu_flat_work_group_size(DIM_M*DIM_N,DIM_M*DIM_N)))
__global__
static void gemm_batched_general_kernel(
    int M, int N, int K, const T alpha,
    const T* const dA_array[], int lda,
    const T* const dB_array[], int ldb,
    const T beta,
          T* const dC_array[], int ldc,
    int batch_count)
{
    int thx = threadIdx.x;     // thread's m position in C
    int thy = threadIdx.y;     // thread's n position in C
    int idt = DIM_M*thy + thx; // thread's number
    int blx = blockIdx.x;      // block's m position
    int bly = blockIdx.y;      // block's n position
    int blz = blockIdx.z;      // block's matrix in the batch
    int thxA = idt % DIM_M_A;  // thread's m position for loading A
    int thyA = idt / DIM_M_A;  // thread's n position for loading A
    int thxB = idt % DIM_M_B;  // thread's m position for loading B
    int thyB = idt / DIM_M_B;  // thread's n position for loading B

    const T* dA = dA_array[blz];
    const T* dB = dB_array[blz];
    T* dC = dC_array[blz];

    __shared__ T sA[BLK_K][BLK_M];  // shared memory for A
    __shared__ T sB[BLK_N][BLK_K];  // shared memory for B
    T rC[BLK_N/DIM_N][BLK_M/DIM_M]; // registers for C

    int a_i_offset = thxA + BLK_M * blx;
    int a_j_offset = thyA;
    int b_i_offset = thxB;
    int b_j_offset = thyB + BLK_N * bly;

    for (int n = 0; n < BLK_N/DIM_N; ++n)
        for (int m = 0; m < BLK_M/DIM_M; ++m)
            rC[n][m] = 0.0;

    int kk = 0;
    for (; kk < K; kk += BLK_K)
    {
        for (int n = 0; n < BLK_K; n += DIM_N_A)
        {
            for (int m = 0; m < BLK_M; m += DIM_M_A)
            {
                int i =  m + a_i_offset;
                int j =  n + kk + a_j_offset;
                if(i < M && j < K)
                    sA[n+thyA][m+thxA] = dA[i + j*lda];
                else
                    sA[n+thyA][m+thxA] = 0.0;
            }
        }


        for (int n = 0; n < BLK_N; n += DIM_N_B)
        {
            for (int m = 0; m < BLK_K; m += DIM_M_B)
            {
                int i =  m + kk + b_i_offset;
                int j =  n + b_j_offset;
                if(i < K && j < N)
                    sB[n+thyB][m+thxB] = dB[i + j*ldb];
                else
                    sB[n+thyB][m+thxB] = 0;
            }
        }

        __syncthreads();

        for (int k = 0; k < BLK_K; ++k)
            for (int n = 0; n < BLK_N/DIM_N; ++n)
                for (int m = 0; m < BLK_M/DIM_M; ++m)
                    rC[n][m] += sA[k][m*DIM_M+thx] * sB[n*DIM_N+thy][k];

        __syncthreads();
    }

    for (int n = 0; n < BLK_N/DIM_N; ++n) {
        for (int m = 0; m < BLK_M/DIM_M; ++m) {
            int coord_dCm = blx*BLK_M + m*DIM_M+thx;
            int coord_dCn = bly*BLK_N + n*DIM_N+thy;
            if(coord_dCn < N && coord_dCm < M)
                dC[coord_dCn*ldc + coord_dCm] = alpha * rC[n][m] + beta * dC[coord_dCn*ldc + coord_dCm]; 
        }
    }
}

// general alpha and beta
template <typename T,
          int DIM_M,
          int DIM_N,
          int BLK_M,
          int BLK_N,
          int BLK_K,
          int DIM_M_A,
          int DIM_N_A,
          int DIM_M_B,
          int DIM_N_B>
__attribute__((amdgpu_flat_work_group_size(DIM_M * DIM_N, DIM_M* DIM_N))) __global__ static void
    gemm_batched_kernel(int    M,
                        int    N,
                        int    K,
                        const T        alpha,
                        const T* const dA_array[],
                        int    lda,
                        const T* const dB_array[],
                        int    ldb,
                        const T        beta,
                        T* const       dC_array[],
                        int    ldc,
                        int    batch_count)
{
    int thx  = threadIdx.x; // thread's m position in C
    int thy  = threadIdx.y; // thread's n position in C
    int idt  = DIM_M * thy + thx; // thread's number
    int blx  = blockIdx.x; // block's m position
    int bly  = blockIdx.y; // block's n position
    int blz  = blockIdx.z; // block's matrix in the batch
    int thxA = idt % DIM_M_A; // thread's m position for loading A
    int thyA = idt / DIM_M_A; // thread's n position for loading A
    int thxB = idt % DIM_M_B; // thread's m position for loading B
    int thyB = idt / DIM_M_B; // thread's n position for loading B

    const T* dA = dA_array[blz];
    const T* dB = dB_array[blz];
    T*       dC = dC_array[blz];

    if(alpha == 0 || K == 0)
    {
        if(beta == 0)
        {
            for(int n = 0; n < BLK_N / DIM_N; ++n)
            {
                for(int m = 0; m < BLK_M / DIM_M; ++m)
                {
                    int coord_dCm                   = blx * BLK_M + m * DIM_M + thx;
                    int coord_dCn                   = bly * BLK_N + n * DIM_N + thy;
                    dC[coord_dCn * ldc + coord_dCm] = 0.0;
                }
            }
        }
        else
        {
            for(int n = 0; n < BLK_N / DIM_N; ++n)
            {
                for(int m = 0; m < BLK_M / DIM_M; ++m)
                {
                    int coord_dCm                   = blx * BLK_M + m * DIM_M + thx;
                    int coord_dCn                   = bly * BLK_N + n * DIM_N + thy;
                    dC[coord_dCn * ldc + coord_dCm] = beta * dC[coord_dCn * ldc + coord_dCm];
                }
            }
        }
    }
    else
    {
        __shared__ T sA[BLK_K][BLK_M]; // shared memory for A
        __shared__ T sB[BLK_N][BLK_K]; // shared memory for B
        T            rC[BLK_N / DIM_N][BLK_M / DIM_M]; // registers for C

        int coord_A = (blx * BLK_M + thyA * lda) + thxA;
        int coord_B = (bly * BLK_N * ldb + thyB * ldb) + thxB;

        for(int n = 0; n < BLK_N / DIM_N; ++n)
            for(int m = 0; m < BLK_M / DIM_M; ++m)
                rC[n][m] = 0.0;

        int kk = 0;
        for(; kk < K; kk += BLK_K)
        {
            for(int n = 0; n < BLK_K; n += DIM_N_A)
                for(int m = 0; m < BLK_M; m += DIM_M_A)
                    sA[n + thyA][m + thxA] = dA[coord_A + (n * lda + m)];

            for(int n = 0; n < BLK_N; n += DIM_N_B)
                for(int m = 0; m < BLK_K; m += DIM_M_B)
                    sB[n + thyB][m + thxB] = dB[coord_B + (n * ldb + m)];

            __syncthreads();

            for(int k = 0; k < BLK_K; ++k)
                for(int n = 0; n < BLK_N / DIM_N; ++n)
                    for(int m = 0; m < BLK_M / DIM_M; ++m)
                        rC[n][m] += sA[k][m * DIM_M + thx] * sB[n * DIM_N + thy][k];

            __syncthreads();

            coord_A += BLK_K * lda;
            coord_B += BLK_K;
        }

        if(beta == 0)
        {
            for(int n = 0; n < BLK_N / DIM_N; ++n)
            {
                for(int m = 0; m < BLK_M / DIM_M; ++m)
                {
                    int coord_dCm                   = blx * BLK_M + m * DIM_M + thx;
                    int coord_dCn                   = bly * BLK_N + n * DIM_N + thy;
                    dC[coord_dCn * ldc + coord_dCm] = alpha * rC[n][m];
                }
            }
        }
        else
        {
            for(int n = 0; n < BLK_N / DIM_N; ++n)
            {
                for(int m = 0; m < BLK_M / DIM_M; ++m)
                {
                    int coord_dCm = blx * BLK_M + m * DIM_M + thx;
                    int coord_dCn = bly * BLK_N + n * DIM_N + thy;
                    dC[coord_dCn * ldc + coord_dCm]
                        = alpha * rC[n][m] + beta * dC[coord_dCn * ldc + coord_dCm];
                }
            }
        }
    }
}


// templated alpha and beta
template <typename T,
          int DIM_M, int DIM_N,
          int BLK_M, int BLK_N, int BLK_K,
          int DIM_M_A, int DIM_N_A,
          int DIM_M_B, int DIM_N_B,
          int alpha, int beta>
__attribute__((amdgpu_flat_work_group_size(DIM_M*DIM_N,DIM_M*DIM_N)))
__global__
static void gemm_batched_kernel(
    int M, int N, int K,
    const T* const dA_array[], int lda,
    const T* const dB_array[], int ldb,
          T* const dC_array[], int ldc,
    int batch_count)
{
    int thx = threadIdx.x;     // thread's m position in C
    int thy = threadIdx.y;     // thread's n position in C
    int idt = DIM_M*thy + thx; // thread's number
    int blx = blockIdx.x;      // block's m position
    int bly = blockIdx.y;      // block's n position
    int blz = blockIdx.z;      // block's matrix in the batch
    int thxA = idt % DIM_M_A;  // thread's m position for loading A
    int thyA = idt / DIM_M_A;  // thread's n position for loading A
    int thxB = idt % DIM_M_B;  // thread's m position for loading B
    int thyB = idt / DIM_M_B;  // thread's n position for loading B

    const T* dA = dA_array[blz];
    const T* dB = dB_array[blz];
    T* dC = dC_array[blz];

    __shared__ T sA[BLK_K][BLK_M];  // shared memory for A
    __shared__ T sB[BLK_N][BLK_K];  // shared memory for B
    T rC[BLK_N/DIM_N][BLK_M/DIM_M]; // registers for C

    int coord_A = (blx*BLK_M     + thyA*lda) + thxA;
    int coord_B = (bly*BLK_N*ldb + thyB*ldb) + thxB;

    for (int n = 0; n < BLK_N/DIM_N; ++n)
        for (int m = 0; m < BLK_M/DIM_M; ++m)
            rC[n][m] = 0.0;

    int kk = 0;
    for (; kk < K; kk += BLK_K)
    {
        for (int n = 0; n < BLK_K; n += DIM_N_A)
            for (int m = 0; m < BLK_M; m += DIM_M_A)
                sA[n+thyA][m+thxA] = dA[coord_A + (n*lda+m)];

        for (int n = 0; n < BLK_N; n += DIM_N_B)
            for (int m = 0; m < BLK_K; m += DIM_M_B)
                sB[n+thyB][m+thxB] = dB[coord_B + (n*ldb+m)];

        __syncthreads();

        for (int k = 0; k < BLK_K; ++k)
            for (int n = 0; n < BLK_N/DIM_N; ++n)
                for (int m = 0; m < BLK_M/DIM_M; ++m)
                    rC[n][m] += sA[k][m*DIM_M+thx] * sB[n*DIM_N+thy][k];

        __syncthreads();

        coord_A += BLK_K*lda;
        coord_B += BLK_K;
    }

    for (int n = 0; n < BLK_N/DIM_N; ++n) {
        for (int m = 0; m < BLK_M/DIM_M; ++m) {
            int coord_dCm = blx*BLK_M + m*DIM_M+thx;
            int coord_dCn = bly*BLK_N + n*DIM_N+thy;
 
            if (alpha == 1 && beta == 1) {
                dC[coord_dCn*ldc + coord_dCm] += rC[n][m];
            }
            else if (alpha ==  1 && beta == -1) {
                dC[coord_dCn*ldc + coord_dCm] = -dC[coord_dCn*ldc + coord_dCm] + rC[n][m];
            }
            else if (alpha == -1 && beta == 0) {
                dC[coord_dCn*ldc + coord_dCm] = -rC[n][m];
            }
            else if (alpha == 1 && beta == 0) {
                dC[coord_dCn*ldc + coord_dCm] = rC[n][m];
            }

        }
    }
}

template <typename T>
void gemm_batched_solution(rocblas_operation trans_a, rocblas_operation trans_b,
                    int m, int n, int k,
                    const T alpha, const T* const dA_array[], int lda,
                                    const T* const dB_array[], int ldb,
                    const T beta,        T* const dC_array[], int ldc,
                    int batch_count, hipStream_t stream)
{
    printf("\n+++gemm_batched_solution+++");
    if((m % 64 == 0) && (n % 64 == 0) && (k % 4 == 0)) 
    {
        printf("   --m 64 --n 64 --k 4   ");
        //m is mult of 64, n is mult of 64, k is mult of 4
        const int dim_m = 16;
        const int dim_n = 16;
        const int blk_m = 64;
        const int blk_n = 64;
        const int blk_k =  4;
        dim3 dimBlock(dim_m, dim_n, 1);
        dim3 dimGrid(m/blk_m, n/blk_n, batch_count);
        if(alpha == 1.0 && beta == 1.0)
        {
            printf("alpha==1  beta==1   ");
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(
                    gemm_batched_kernel<
                        T,
                        dim_m, dim_n,
                        blk_m, blk_n, blk_k,
                        blk_m, blk_k,
                        blk_k, blk_n,
                        1, 1>),
                dimGrid, dimBlock, 0, stream,
                m, n, k,
                dA_array, lda,
                dB_array, ldb,
                dC_array, ldc,
                batch_count);
        }
        else if(alpha == 1.0 && beta == -1.0)
        {
            printf("alpha==1  beta==-1   ");
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(
                    gemm_batched_kernel<
                        T,
                        dim_m, dim_n,
                        blk_m, blk_n, blk_k,
                        blk_m, blk_k,
                        blk_k, blk_n,
                        1, -1>),
                dimGrid, dimBlock, 0, stream,
                m, n, k,
                dA_array, lda,
                dB_array, ldb,
                dC_array, ldc,
                batch_count);
        }
        else if(alpha == 1.0 && beta == 0.0)
        {
            printf("alpha==1  beta==0   ");
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(
                    gemm_batched_kernel<
                        T,
                        dim_m, dim_n,
                        blk_m, blk_n, blk_k,
                        blk_m, blk_k,
                        blk_k, blk_n,
                        1, 0>),
                dimGrid, dimBlock, 0, stream,
                m, n, k,
                dA_array, lda,
                dB_array, ldb,
                dC_array, ldc,
                batch_count);
        }
        else if(alpha == -1.0 && beta == 0.0)
        {
            printf("alpha==-1  beta==0   ");
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(
                    gemm_batched_kernel<
                        T,
                        dim_m, dim_n,
                        blk_m, blk_n, blk_k,
                        blk_m, blk_k,
                        blk_k, blk_n,
                        -1, 0>),
                dimGrid, dimBlock, 0, stream,
                m, n, k,
                dA_array, lda,
                dB_array, ldb,
                dC_array, ldc,
                batch_count);
        }
        else
        {
            printf("general alpha  beta   ");
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(
                    gemm_batched_kernel<
                        T,
                        dim_m, dim_n,
                        blk_m, blk_n, blk_k,
                        blk_m, blk_k,
                        blk_k, blk_n>),
                dimGrid, dimBlock, 0, stream,
                m, n, k, alpha,
                dA_array, lda,
                dB_array, ldb,
                beta,
                dC_array, ldc,
                batch_count);
        }
    }
    else if((m % 32 == 0) && (n % 32 == 0) && (k % 8 == 0)) 
    {
        printf("   --m 32 --n 32 --k 8   ");
        // m is mult of 32, n is mult of 32, k is mult of 8
        const int dim_m = 16;
        const int dim_n = 16;
        const int blk_m = 32;
        const int blk_n = 32;
        const int blk_k =  8;
        dim3 dimBlock(dim_m, dim_n, 1);
        dim3 dimGrid(m/blk_m, n/blk_n, batch_count);
        if(alpha == 1.0 && beta == 1.0)
        {
            printf("alpha==1  beta==1   ");
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(
                    gemm_batched_kernel<
                        T,
                        dim_m, dim_n,
                        blk_m, blk_n, blk_k,
                        blk_m, blk_k,
                        blk_k, blk_n,
                        1, 1>),
                dimGrid, dimBlock, 0, stream,
                m, n, k,
                dA_array, lda,
                dB_array, ldb,
                dC_array, ldc,
                batch_count);
        }
        else if(alpha == 1.0 && beta == -1.0)
        {
            printf("alpha==1  beta==-1   ");
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(
                    gemm_batched_kernel<
                        T,
                        dim_m, dim_n,
                        blk_m, blk_n, blk_k,
                        blk_m, blk_k,
                        blk_k, blk_n,
                        1, -1>),
                dimGrid, dimBlock, 0, stream,
                m, n, k,
                dA_array, lda,
                dB_array, ldb,
                dC_array, ldc,
                batch_count);
        }
        else if(alpha == 1.0 && beta == 0.0)
        {
            printf("alpha==1  beta==0   ");
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(
                    gemm_batched_kernel<
                        T,
                        dim_m, dim_n,
                        blk_m, blk_n, blk_k,
                        blk_m, blk_k,
                        blk_k, blk_n,
                        1, 0>),
                dimGrid, dimBlock, 0, stream,
                m, n, k,
                dA_array, lda,
                dB_array, ldb,
                dC_array, ldc,
                batch_count);
        }
        else if(alpha == -1.0 && beta == 0.0)
        {
            printf("alpha==-1  beta==0   ");
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(
                    gemm_batched_kernel<
                        T,
                        dim_m, dim_n,
                        blk_m, blk_n, blk_k,
                        blk_m, blk_k,
                        blk_k, blk_n,
                        -1, 0>),
                dimGrid, dimBlock, 0, stream,
                m, n, k,
                dA_array, lda,
                dB_array, ldb,
                dC_array, ldc,
                batch_count);
        }
        else
        {
            printf("general alpha  beta   ");
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(
                    gemm_batched_kernel<
                        T,
                        dim_m, dim_n,
                        blk_m, blk_n, blk_k,
                        blk_m, blk_k,
                        blk_k, blk_n>),
                dimGrid, dimBlock, 0, stream,
                m, n, k, alpha,
                dA_array, lda,
                dB_array, ldb,
                beta,
                dC_array, ldc,
                batch_count);
        }
    }
    else
    {
        const int dim_m = 16;
        const int dim_n = 16;
        const int blk_m = 32;
        const int blk_n = 32;
        const int blk_k =  8;
        dim3 dimBlock(dim_m, dim_n, 1);
        dim3 dimGrid(((m-1)/blk_m)+1, ((n-1)/blk_n)+1, batch_count);
        printf("general alpha  beta  m  n  k    ");
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(
                gemm_batched_general_kernel<
                    T,
                    dim_m, dim_n,
                    blk_m, blk_n, blk_k,
                    blk_m, blk_k,
                    blk_k, blk_n>),
            dimGrid, dimBlock, 0, stream,
            m, n, k, alpha,
            dA_array, lda,
            dB_array, ldb,
            beta,
            dC_array, ldc,
            batch_count);
    }
}

//----------------------------------------------------------------------------
template
void gemm_batched_solution<float>(
    rocblas_operation trans_a, rocblas_operation trans_b,
    int m, int n, int k,
    const float alpha,
    const float* const dA_array[], int lda,
    const float* const dB_array[], int ldb,
    const float beta,
    float* const dC_array[], int ldc,
    int batch_count, hipStream_t stream);

template
void gemm_batched_solution<double>(
    rocblas_operation trans_a, rocblas_operation trans_b,
    int m, int n, int k,
    const double alpha,
    const double* const dA_array[], int lda,
    const double* const dB_array[], int ldb,
    const double beta,
    double* const dC_array[], int ldc,
    int batch_count, hipStream_t stream);

/*
template
void gemm_batched_solution<std::complex<float>>(
    int m, int n, int k,
    const std::complex<float>* alpha,
    const std::complex<float>* const dA_array[], int lda,
    const std::complex<float>* const dB_array[], int ldb,
    const std::complex<float>* beta,
    std::complex<float>* const dC_array[], int ldc,
    int batch_count, hipStream_t stream);

template
void gemm_batched_solution<std::complex<double>>(
    int m, int n, int k,
    const std::complex<double> alpha,
    const std::complex<double>* const dA_array[], int lda,
    const std::complex<double>* const dB_array[], int ldb,
    const std::complex<double> beta,
    std::complex<double>* const dC_array[], int ldc,
    int batch_count, hipStream_t stream);
*/
