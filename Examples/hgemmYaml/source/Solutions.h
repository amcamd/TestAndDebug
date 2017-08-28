#include "TensileTypes.h"
#include "Kernels.h"
#include "SolutionHelper.h"
#include "Tools.h"
// Header

hipError_t Cij_Aik_Bkj_HB_MT032x032x08_(
    TensileHalf * dataC,
    const TensileHalf * dataA,
    const TensileHalf * dataB,
    TensileHalf alpha,
    TensileHalf beta,
    unsigned int offsetC,
    unsigned int offsetA,
    unsigned int offsetB,
    unsigned int strideC1J,
    unsigned int strideA1K,
    unsigned int strideB1J,
    unsigned int sizeI,
    unsigned int sizeJ,
    unsigned int sizeK,
    hipStream_t stream,
    unsigned int numInputEvents,
    hipEvent_t * inputEvents,
    hipEvent_t * outputEvent);


