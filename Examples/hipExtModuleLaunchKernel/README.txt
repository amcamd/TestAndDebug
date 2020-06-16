
The aim of this code is to show the use of any-order launch. It is based on the 
code at: /opt/rocm/hip/samples/0_Intro/module_api


Below shows that flags=1 with hipExtModuleLaunchKernel improves performance.


work_kernel has runtime of 116us

Comparing default stream to created stream:
default stream
empty_kernel, flags = 0: LEN, batch_count, seconds = 256, 1000, 0.0078024
empty_kernel, flags = 1: LEN, batch_count, seconds = 256, 1000, 0.00687898
 work_kernel, flags = 0: LEN, batch_count, seconds = 256, 1000, 0.131641
 work_kernel, flags = 1: LEN, batch_count, seconds = 256, 1000, 0.120073


created stream
empty_kernel, flags = 0: LEN, batch_count, seconds = 256, 1000, 0.0077224
empty_kernel, flags = 1: LEN, batch_count, seconds = 256, 1000, 0.00688674
 work_kernel, flags = 0: LEN, batch_count, seconds = 256, 1000, 0.131606
 work_kernel, flags = 1: LEN, batch_count, seconds = 256, 1000, 0.120286





Below is from  /opt/rocm/hip/include/hip$ vim hip_ext.h



/**
 * @brief launches kernel f with launch parameters and shared memory on stream with arguments passed
 to kernelparams or extra
 *
 * @param [in[ f     Kernel to launch.
 * @param [in] gridDimX  X grid dimension specified in work-items
 * @param [in] gridDimY  Y grid dimension specified in work-items
 * @param [in] gridDimZ  Z grid dimension specified in work-items
 * @param [in] blockDimX X block dimensions specified in work-items
 * @param [in] blockDimY Y grid dimension specified in work-items
 * @param [in] blockDimZ Z grid dimension specified in work-items
 * @param [in] sharedMemBytes Amount of dynamic shared memory to allocate for this kernel.  The
 kernel can access this with HIP_DYNAMIC_SHARED.
 * @param [in] stream Stream where the kernel should be dispatched.  May be 0, in which case th
 default stream is used with associated synchronization rules.
 * @param [in] kernelParams
 * @param [in] extra     Pointer to kernel arguments.   These are passed directly to the kernel and
 must be in the memory layout and alignment expected by the kernel.
 * @param [in] startEvent  If non-null, specified event will be updated to track the start time of
 the kernel launch.  The event must be created before calling this API.
 * @param [in] stopEvent   If non-null, specified event will be updated to track the stop time of
 the kernel launch.  The event must be created before calling this API.
 *
 * @returns hipSuccess, hipInvalidDevice, hipErrorNotInitialized, hipErrorInvalidValue
 *
 * @warning kernellParams argument is not yet implemented in HIP. Please use extra instead. Please
 refer to hip_porting_driver_api.md for sample usage.
 * HIP/ROCm actually updates the start event when the associated kernel completes.
 */
HIP_PUBLIC_API
hipError_t hipExtModuleLaunchKernel(hipFunction_t f, uint32_t globalWorkSizeX,
                                    uint32_t globalWorkSizeY, uint32_t globalWorkSizeZ,
                                    uint32_t localWorkSizeX, uint32_t localWorkSizeY,
                                    uint32_t localWorkSizeZ, size_t sharedMemBytes,
                                    hipStream_t hStream, void** kernelParams, void** extra,
                                    hipEvent_t startEvent = nullptr,
                                    hipEvent_t stopEvent = nullptr,
                                    uint32_t flags = 0);


batched, flags = 1
transA,transB,M,N,K,alpha,lda,ldb,beta,ldc,Batch_Count,rocblas-Gflops,us
N,N,192,192,32,1,192,128,0,192,1000,215.863,10929.6

transA,transB,M,N,K,alpha,lda,ldb,beta,ldc,Batch_Count,rocblas-Gflops,us
N,N,1536,1536,32,1,1536,128,0,1536,1000,1918.349,78710.9


batched, flags = 0
work group = 64, 1, 1    work items = 768, 12, 1
Cijk_Ailk_Bljk_SB_MT16x16x16_SE_AF0EM1_AMAS2_ASEM1_BL0_DTL0_EPS0_FL0_GRVW2_GSU1_ISA000_K1_KLS_LPA0_LPB0_NLCA1_NLCB1_PGR1_PLR1_TT2_2_USFGRO0_VAW1_VW2_WG8_8_1_WGM1
transA,transB,M,N,K,alpha,lda,ldb,beta,ldc,Batch_Count,rocblas-Gflops,us
N,N,192,192,32,1,192,128,0,192,1000,208.3264,11325

work group = 256, 1, 1    work items = 6144, 24, 1
Cijk_Ailk_Bljk_SB_MT64x64x16_SN_AF0EM1_AMAS2_ASEM1_BL0_DTL0_EPS0_FL0_GRVW4_GSU1_ISA000_K1_KLS_LPA0_LPB0_NLCA1_NLCB1_PGR1_PLR1_TT4_4_USFGRO0_VAW1_VW4_WG16_16_1_WGM1
transA,transB,M,N,K,alpha,lda,ldb,beta,ldc,Batch_Count,rocblas-Gflops,us
N,N,1536,1536,32,1,1536,128,0,1536,1000,1917.279,78754.8
