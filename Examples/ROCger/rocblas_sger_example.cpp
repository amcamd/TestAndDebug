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

void usage(char *argv[])
{
    printf("Usage: %s\n", argv[0]);
    printf(" -m<ger m, default %d>\n", M_DIM);
    printf(" -n<ger n, default %d>\n", N_DIM);
    printf(" -a<ger lda, default %d>\n", M_DIM);
    printf(" -x<ger incx, default %d>\n", 1);
    printf(" -y<ger incy, default %d>\n", 1);
    exit (8);
}


int parse_args(int argc, char *argv[], int &M, int &N, int &lda, int &incx, int &incy)
{
    while (argc > 1)
    {
       if (argv[1][0] == '-')
       {
           switch (argv[1][1])
           {
               case 'm':
                   M = atoi(&argv[1][2]);
                   break;
               case 'n':
                   N = atoi(&argv[1][2]);
                   break;
               case 'a':
                   lda = atoi(&argv[1][2]);
                   break;
               case 'x':
                   incx = atoi(&argv[1][2]);
                   break;
               case 'y':
                   incy = atoi(&argv[1][2]);
                   break;
               default:
                   printf("Wrong Argument: %s\n", argv[1]);
                   return (1);
            }
        }
        else
        {
           printf("Wrong Argument: %s\n", argv[1]);
           return (1);
        }
        ++argv;
        --argc;
    }
    return (0);
}


int main(int argc, char *argv[]) {

    rocblas_int M = M_DIM;
    rocblas_int N = N_DIM;
    rocblas_int incx = 1;
    rocblas_int incy = 1;
    rocblas_int lda = M_DIM;
    float alpha = 1.0;
    if( parse_args(argc, argv, M, N, lda, incx, incy)) {
        usage(argv);
    }
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
    if ( lda >= M ){
        for( int i = 0; i < M; ++i )
        {
            for(int j = 0; j < N; ++j)
            {
                hA[i + j * lda] = rand() % 10 + 1;
            }
        }
    }

    // save a copy in hA
    hB = hA;

    hipMemcpy(dx, hx.data(), sizeof(float) * M * incx, hipMemcpyHostToDevice);
    hipMemcpy(dy, hy.data(), sizeof(float) * N * incy, hipMemcpyHostToDevice);
    hipMemcpy(dA, hA.data(), sizeof(float) * N * lda,  hipMemcpyHostToDevice);

    rocblas_status status = rocblas_sger(handle, M, N, &alpha, dx, incx, dy, incy, dA, lda);
    if (status != rocblas_status_success) 
    {
        printf("rocblas_sger failed and returned rocblas_status = %d\n", status);
        return 1;
    }

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
