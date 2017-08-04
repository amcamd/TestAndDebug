 /**
 -- (C) Copyright 2013 King Abdullah University of Science and Technology
  Authors:
  Ahmad Abdelfattah (ahmad.ahmad@kaust.edu.sa)
  David Keyes (david.keyes@kaust.edu.sa)
  Hatem Ltaief (hatem.ltaief@kaust.edu.sa)

  Redistribution  and  use  in  source and binary forms, with or without
  modification,  are  permitted  provided  that the following conditions
  are met:

  * Redistributions  of  source  code  must  retain  the above copyright
    notice,  this  list  of  conditions  and  the  following  disclaimer.
  * Redistributions  in  binary  form must reproduce the above copyright
    notice,  this list of conditions and the following disclaimer in the
    documentation  and/or other materials provided with the distribution.
  * Neither  the  name of the King Abdullah University of Science and
    Technology nor the names of its contributors may be used to endorse 
    or promote products derived from this software without specific prior 
    written permission.

  THIS  SOFTWARE  IS  PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
  ``AS IS''  AND  ANY  EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
  LIMITED  TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
  A  PARTICULAR  PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
  HOLDERS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
  SPECIAL,  EXEMPLARY,  OR  CONSEQUENTIAL  DAMAGES  (INCLUDING,  BUT NOT
  LIMITED  TO,  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
  DATA,  OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
  THEORY  OF  LIABILITY,  WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  (INCLUDING  NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF  THIS  SOFTWARE,  EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
**/

*****************************************************************************/

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
////        syhemvl_special_d<float, ssymv_bs, thread_x, thread_y, elements_per_thread><<<dimGrid, dimBlock, 0, stream>>> ( m, alpha, dA, lda, dX, incx, beta, dY, incy);
////        syhemvl_special_nd<float, ssymv_bs, thread_x, thread_y, elements_per_thread><<<dimGrid_, dimBlock, 0, stream>>> ( m, alpha, dA, lda, dX, incx, beta, dY, incy);
		}
		else
		{
////        syhemvl_generic_d<float, ssymv_bs, thread_x, thread_y, elements_per_thread><<<dimGrid, dimBlock, 0, stream>>> ( m, alpha, dA, lda, dX, incx, beta, dY, incy, mod);
////        syhemvl_generic_nd<float, ssymv_bs, thread_x, thread_y, elements_per_thread><<<dimGrid_, dimBlock, 0, stream>>> ( m, alpha, dA, lda, dX, incx, beta, dY, incy, mod);
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
