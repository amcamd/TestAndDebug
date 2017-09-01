#include "Solutions.h"
// Header

hipError_t Cij_Aik_Bkj_HB_MT128x128x08_(
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
    hipEvent_t * outputEvent) {

  /* kernels */
  const unsigned int numKernels = 1; // 1 or 4

  /* index assignments */
  const unsigned int indexD0 = 0;
  const unsigned int indexD1 = 1;
  const unsigned int indexDU = 2;

  /* num kernels */
  unsigned int numEnqueues[numKernels] = { 1 };

  /* grid sizes */
  const unsigned int workDim = 3;
  const unsigned int threadTile[2] = { 8, 8 };
  const unsigned int groupSize[2] = { 16, 16 };
  size_t localWorkSize[3] = { 256, 1, 1 };
  size_t globalWorkSize[numKernels][3];
  globalWorkSize[0][2] = 1;
  unsigned int sizeOfC0 = sizeI;
  unsigned int sizeOfC1 = sizeJ;
  unsigned int macroTile0 = static_cast<unsigned int>(groupSize[0] * threadTile[0]);
  unsigned int macroTile1 = static_cast<unsigned int>(groupSize[1] * threadTile[1]);
  unsigned int totalWorkGroups0 = sizeOfC0 / macroTile0;
  unsigned int totalWorkGroups1 = sizeOfC1 / macroTile1;
  // b/c single kernel, add extra work-group here if edge needed
  if (totalWorkGroups0*macroTile0 < sizeOfC0) { totalWorkGroups0++; }
  if (totalWorkGroups1*macroTile1 < sizeOfC1) { totalWorkGroups1++; }
  globalWorkSize[0][0] = totalWorkGroups0;
  globalWorkSize[0][1] = totalWorkGroups1;

  /* offsets */
  unsigned int offsets[numKernels][1][3];
  offsets[0][0][0] = offsetC; // tensorC
  offsets[0][0][1] = offsetA; // tensorA
  offsets[0][0][2] = offsetB; // tensorB

  /* index sizes */
  unsigned int sizes[numKernels][1][3];
  sizes[0][0][0] = sizeI;
  sizes[0][0][1] = sizeJ;
  sizes[0][0][2] = sizeK;

  TensileStatus status;


  /* kernel 0: Cij_Aik_Bkj_HB_MT128x128x08_K1 */
  unsigned int kernelIdx = 0;
  for (unsigned int enqueueIdx = 0; enqueueIdx < numEnqueues[0]; enqueueIdx++) {
    if( inputEvents != NULL )
      hipEventRecord(inputEvents[enqueueIdx], stream );
    try {
      hipLaunchKernel(
        HIP_KERNEL_NAME(Cij_Aik_Bkj_HB_MT128x128x08_K1),
        dim3(globalWorkSize[kernelIdx][0], globalWorkSize[kernelIdx][1], globalWorkSize[kernelIdx][2]),
        dim3(localWorkSize[0], localWorkSize[1], localWorkSize[2]),
        0, // groupMemBytes
        stream,
        dataC,
        dataA,
        dataB,
        alpha,
        beta,
        offsets[kernelIdx][enqueueIdx][0],
        offsets[kernelIdx][enqueueIdx][1],
        offsets[kernelIdx][enqueueIdx][2],
        strideC1J,
        strideA1K,
        strideB1J,
        sizes[kernelIdx][enqueueIdx][0],
        sizes[kernelIdx][enqueueIdx][1],
        sizes[kernelIdx][enqueueIdx][2]
    );
      } catch (const std::exception& e) {
#ifdef DEBUG
        std::cerr << e.what() << std::endl;
#endif
        return tensileStatusFailure;
      }
      if( outputEvent != NULL )
        hipEventRecord(outputEvent[enqueueIdx], stream );
  }

  return tensileStatusSuccess;
}

/* Solution Parameters
  ProblemType: Cij_Aik_Bkj_HB
  PrefetchGlobalRead: False
  UnrollMemFence: False
  GlobalRead2A: True
  LoopDoWhile: False
  LocalWrite2A: True
  LocalWrite2B: True
  LdsPad: 0
  MaxOccupancy: 10
  MacroTileShapeMax: 4
  Valid: True
  GlobalReadCoalesceVectorB: True
  DepthU: 8
  GlobalReadCoalesceVectorA: True
  GlobalLoadVectorWidthA: 4
  MacroTileShapeMin: 1
  LocalRead2A: True
  LocalRead2B: True
  PVA: 2
  LdsNumElements: 2175.0
  NumLoadsA: 1
  NumLoadsB: 1
  EdgeType: ShiftPtr
  NumThreads: 256
  GlobalLoadVectorWidthB: 4
  ThreadTile1: 8
  ThreadTile0: 8
  VectorWidth: 8
  NumVectorsPerThread: 8
  PVB: 2
  GlobalSplitUWorkGroupMappingRoundRobin: True
  WorkGroup: [16, 16, 1]
  LoopTail: False
  ProblemType: Cij_Aik_Bkj_HB
  LocalSplitU: 1
  NumLoadsPerpendicularA: 1
  AssignedProblemIndependentDerivedParameters: True
  GlobalRead2B: True
  WorkGroupMapping: 1
  LoopUnroll: 8
  PrefetchLocalRead: False
  SubGroup0: 16
  SubGroup1: 16
  WorkGroupMappingType: B
  AssignedDerivedParameters: True
  GlobalReadCoalesceGroupB: True
  GlobalReadCoalesceGroupA: True
  MacroTile1: 128
  GlobalSplitU: 1
  NumLoadsCoalescedA: 1
  NumLoadsCoalescedB: 1
  GlobalSplitUSummationAssignmentRoundRobin: True
  ThreadTile: [8, 8]
  MacroTile0: 128
  LdsOffsetB: 1151.0
  NumLoadsPerpendicularB: 1
*/

