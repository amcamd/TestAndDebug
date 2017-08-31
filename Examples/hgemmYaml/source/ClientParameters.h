// Header

#include "Solutions.h"

typedef enum {
    enum_float,
    enum_double,
    enum_TensileComplexFloat,
    enum_TensileComplexDouble
#ifdef Tensile_ENABLE_HALF
    ,enum_TensileHalf
#endif
} DataTypeEnum;

const char indexChars[19] = "IJKLMNOPQRSTUVWXYZ";
unsigned int functionIdx;
unsigned int dataTypeIdx;
unsigned int problemTypeIdx;

/* data types */
const unsigned int numDataTypes = 1;
const DataTypeEnum dataTypeEnums[numDataTypes] = { enum_TensileHalf };
const unsigned int bytesPerElement[numDataTypes] = { 2 };
const unsigned int numFlopsPerMac[numDataTypes] = { 2 };
#define Tensile_DATA_TYPE_TENSILEHALF
/* problem types */
const unsigned int numProblemTypes = 1;
const unsigned int numIndicesC[numProblemTypes] = { 2 };
const unsigned int numIndicesAB[numProblemTypes] = { 2 };
const unsigned int maxNumIndicesAB = 2;
const unsigned int indexAssignmentsA[numProblemTypes][maxNumIndicesAB] = {
  { 0, 2 }
};
const unsigned int indexAssignmentsB[numProblemTypes][maxNumIndicesAB] = {
  { 2, 1 }
};
bool useBeta[numProblemTypes] = { true };
const bool complexConjugateA[numProblemTypes] = { false };
const bool complexConjugateB[numProblemTypes] = { false };

const unsigned int maxNumIndices = 3;
const unsigned int totalIndices[numProblemTypes] = { 3 };
const unsigned int numProblems = 1;
const unsigned int problemSizes[numProblems][3] = {
  {  64,  64,  64}};
/* problem sizes */
size_t maxSizeC = 4096;
size_t maxSizeA = 4096;
size_t maxSizeB = 4096;
//size_t maxSizeC = 1024;
//size_t maxSizeA = 1024;
//size_t maxSizeB = 1024;

/* current problem size */

/* solutions */
const unsigned int numSolutions = 1;
float solutionPerf[numProblems][numSolutions]; // milliseconds

typedef TensileStatus (*SolutionFunctionPointer)(
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

const SolutionFunctionPointer solutions[numSolutions] = {
  Cij_Aik_Bkj_HB_MT032x032x08_
 };

const char *solutionNames[numSolutions] = {
  "Cij_Aik_Bkj_HB_MT032x032x08_"
 };

/* runtime structures */
TensileStatus status;
hipStream_t stream;
int deviceIdx = 0;

void *deviceC;
void *deviceA;
void *deviceB;

/* benchmarking parameters */
const bool measureKernelTime = false;
const unsigned int numEnqueuesPerSync = 1;
const unsigned int numSyncsPerBenchmark = 2;
unsigned int numElementsToValidate = 0xFFFFFFFF;
unsigned int validationMaxToPrint = 4;
bool validationPrintValids = false;
size_t validationStride;
unsigned int dataInitTypeC = 3;
unsigned int dataInitTypeAB = 3;

/* generated call to reference */
template<typename DataType>
TensileStatus generatedCallToReferenceCPU(
    const unsigned int *sizes,
    DataType *referenceC,
    DataType *initialA,
    DataType *initialB,
    DataType alpha,
    DataType beta) {
  return tensileReferenceCPU(
      referenceC,
      initialA,
      initialB,
      alpha,
      beta,
      totalIndices[problemTypeIdx],
      sizes,
      numIndicesC[problemTypeIdx],
      numIndicesAB[problemTypeIdx],
      indexAssignmentsA[problemTypeIdx],
      indexAssignmentsB[problemTypeIdx],
      complexConjugateA[problemTypeIdx],
      complexConjugateB[problemTypeIdx],
      validationStride );
};

/* generated call to solution */
template<typename DataType>
TensileStatus generatedCallToSolution(
    unsigned int solutionIdx,
    const unsigned int *sizes,
    DataType alpha,
    DataType beta, 
    unsigned int numEvents = 0, 
    hipEvent_t *startEvent = NULL,
    hipEvent_t *stopEvent = NULL ) {
  // calculate parameters assuming packed data
  unsigned int strideC0I = 1;
  unsigned int strideC1J = 1*sizes[0];
  unsigned int strideA0I = 1;
  unsigned int strideA1K = 1*sizes[0];
  unsigned int strideB0K = 1;
  unsigned int strideB1J = 1*sizes[2];
  unsigned int sizeI = sizes[0];
  unsigned int sizeJ = sizes[1];
  unsigned int sizeK = sizes[2];

  // call solution function
  return solutions[solutionIdx]( static_cast<TensileHalf *>(deviceC), static_cast<TensileHalf *>(deviceA), static_cast<TensileHalf *>(deviceB),
      alpha,
      beta,
      0, 0, 0, // offsets
      strideC1J,
      strideA1K,
      strideB1J,
      sizeI,
      sizeJ,
      sizeK,
      stream,
      numEvents, startEvent, stopEvent); // events
};

/* results file name */
const char *resultsFileName = "/home/achapman/repos/Tensile/build4/1_BenchmarkProblems/Cij_Aik_Bkj_HB_00/00_Final/source/../../Data/00_Final.csv";
