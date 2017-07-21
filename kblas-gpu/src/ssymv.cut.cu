
#include "syhemv_core.cuh"

#if(SM >= 30)

#define ssymv_upper_bs 	(64)
#define ssymv_upper_ty 	(4)
#define ssymv_upper_by	(2)

#define ssymv_lower_bs 	(64)
#define ssymv_lower_ty 	(4)
#define ssymv_lower_by	(2)

#else

#define ssymv_upper_bs 	(32)
#define ssymv_upper_ty 	(8)
#define ssymv_upper_by	(2)

#define ssymv_lower_bs 	(32)
#define ssymv_lower_ty 	(4)
#define ssymv_lower_by	(2)

#endif


int kblas_ssymv_driver( char uplo, 
						int m, float alpha, float *dA, int lda, 
						float *dX, int incx, 
						float  beta, float *dY, int incy, 
						cudaStream_t stream)
{
	// handle the case when incx and/or incy is -ve
	if(incx < 0) dX -= (m-1) * incx;
	if(incy < 0) dY -= (m-1) * incy;
	
	if(uplo == 'U' || uplo == 'u')
	{
	}
	else if(uplo == 'L' || uplo == 'l')
	{
		/** configuration params **/
		const int ssymv_bs = ssymv_lower_bs;
		const int thread_x = ssymv_bs;
		const int thread_y = ssymv_lower_ty;
		const int elements_per_thread = (ssymv_bs/(2*thread_y)) ;
		/** end configuration params **/
		
		int mod = m % ssymv_bs;
		int blocks = m / ssymv_bs + (mod != 0);
		dim3 dimBlock(thread_x, thread_y);
		dim3 dimGrid(blocks,1);
		dim3 dimGrid_(blocks,ssymv_lower_by);

		if(mod == 0)
		{
			syhemvl_special_d <float, ssymv_bs, thread_x, thread_y, elements_per_thread> <<<dimGrid , dimBlock, 0, stream>>> ( m, alpha, dA, lda, dX, incx, beta, dY, incy);
			syhemvl_special_nd<float, ssymv_bs, thread_x, thread_y, elements_per_thread> <<<dimGrid_, dimBlock, 0, stream>>> ( m, alpha, dA, lda, dX, incx, beta, dY, incy);
		}
		else
		{
		  	syhemvl_generic_d <float, ssymv_bs, thread_x, thread_y, elements_per_thread> <<<dimGrid , dimBlock, 0, stream>>> ( m, alpha, dA, lda, dX, incx, beta, dY, incy, mod);
			syhemvl_generic_nd<float, ssymv_bs, thread_x, thread_y, elements_per_thread> <<<dimGrid_, dimBlock, 0, stream>>> ( m, alpha, dA, lda, dX, incx, beta, dY, incy, mod);
		}
	}
	else{printf("Upper/Lower mode %c is not supported \n", uplo); return -1;}	
	return 0;
}


extern "C"
int kblas_ssymv( char uplo, 
				 int m, float alpha, float *dA, int lda, 
				float *dX, int incx, 
				float  beta, float *dY, int incy)
{
	return kblas_ssymv_driver( uplo, m, alpha, dA, lda, dX, incx, beta, dY, incy, 0);
}

extern "C"
int kblas_ssymv_async( char uplo, 
						int m, float alpha, float *dA, int lda, 
						float *dX, int incx, 
						float  beta, float *dY, int incy, 
						cudaStream_t stream)
{
	return kblas_ssymv_driver( uplo, m, alpha, dA, lda, dX, incx, beta, dY, incy, stream);
}
