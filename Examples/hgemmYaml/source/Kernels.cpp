#include "Kernels.h"
// Header


  /******************************************/
  /* Function Prefix                        */
  /******************************************/

/* tile parameters */
#define NUM_THREADS  64
#define SG0I 8
#define SG1J 8
#define TT0I 4
#define TT1J 4
#define MT0I (SG0I*TT0I)
#define MT1J (SG1J*TT1J)
#define VECTOR_WIDTH 4
#define GLOBAL_LOAD_VECTOR_WIDTH_A 4
#define GLOBAL_LOAD_VECTOR_WIDTH_B 4

/* DepthU parameters*/
#define CPSV (NUM_THREADS / MT0I * VECTOR_WIDTH)
#define LOCAL_SPLITU 1
#define UNROLL 8
#define LOCAL_DEPTHU (LOCAL_SPLITU*UNROLL)

/* other */
#define PAD 0
#define WORK_GROUP_MAPPING 1

/* num loads parallel and perpendicular to coalesced */
#define NLCA 1
#define NLCB 1
#define NLPA 1
#define NLPB 1

/* load sizes parallel and perpendicular to coalesced */
#define LSCA (MT0I/NLCA)
#define LSPA (LOCAL_DEPTHU/NLPA)
#define LSCB (LOCAL_DEPTHU/NLCB)
#define LSPB (MT1J/NLPB)
#define LVCA (LSCA/GLOBAL_LOAD_VECTOR_WIDTH_A)
#define LVCB (LSCB/GLOBAL_LOAD_VECTOR_WIDTH_B)
#define LVPA (LSPA/GLOBAL_LOAD_VECTOR_WIDTH_A)
#define LVPB (LSPB/GLOBAL_LOAD_VECTOR_WIDTH_B)
#define LDS_OFFSET_B 383
#define LDS_NUM_ELEMENTS 1638

#ifndef Z_ORDER_FUNCTIONS
#define Z_ORDER_FUNCTIONS
__device__ void z_order(
    unsigned int *z0, // 16-bits output
    unsigned int *z1, // 16-bits output
    unsigned int serial ) { // 32-bits input
  *z0 = serial;
  *z1 = (serial >> 1);
  *z0 &= 0x55555555;
  *z1 &= 0x55555555;
  *z0 |= ( (*z0) >> 1 );
  *z1 |= ( (*z1) >> 1 );
  *z0 &= 0x33333333;
  *z1 &= 0x33333333;
  *z0 |= ( (*z0) >> 2 );
  *z1 |= ( (*z1) >> 2 );
  *z0 &= 0x0f0f0f0f; 
  *z1 &= 0x0f0f0f0f;
  *z0 |= ( (*z0) >> 4 );
  *z1 |= ( (*z1) >> 4 );
  *z0 &= 0x00ff00ff;
  *z1 &= 0x00ff00ff;
  *z0 |= ( (*z0) >> 8 );
  *z1 |= ( (*z1) >> 8 );
  *z0 &= 0x0000ffff;
  *z1 &= 0x0000ffff;
}

__device__ unsigned int round_down_power_of_2( unsigned int d0, unsigned int d1) {
  unsigned int pow2 = min(d0, d1);
  pow2 = pow2 | (pow2 >> 1);
  pow2 = pow2 | (pow2 >> 2);
  pow2 = pow2 | (pow2 >> 4);
  pow2 = pow2 | (pow2 >> 8);
  pow2 = pow2 | (pow2 >> 16);
  pow2 = pow2 - (pow2 >> 1);
  return pow2;
}

__device__ void generalized_z_order(
    unsigned int *z0,
    unsigned int *z1,
    unsigned int d0,
    unsigned int d1,
    unsigned int maxPow2,
    unsigned int max0,
    unsigned int max1 ) {
  if (! maxPow2) maxPow2 = round_down_power_of_2( max0, max1 );
  // determine which tile wg is in and relative coord in tile
  unsigned int offset0 = 0; // coord of tile
  unsigned int offset1 = 0; // coord of tile
  unsigned int start0 = 0;
  unsigned int start1 = 0;
  unsigned int tile = maxPow2;
  unsigned int tilem1 = tile - 1;
  for ( unsigned int i = 0; i < 16; i++ ) {
    start0 = d0 & ~tilem1; // (d0 / tile) * tile;
    start1 = d1 & ~tilem1; // (d1 / tile) * tile;
    offset0 |= start0; // +=
    offset1 |= start1;
    d0 &= ~start0; // -=
    d1 &= ~start1;
    unsigned int end0 = start0 + tile; // cant be | b/c evals to 0+4->4 or 4+4->8
    unsigned int end1 = start1 + tile;
    if ( end0 <= max0 && end1 <= max1 ) break; // both end and max can be non-pow2
    max0 -= start0; // cant be &~ b/c max0 doesnt necessarily have multiple of start0 to turn off
    max1 -= start1;
    tile >>= 1;
    tilem1 >>= 1;
  }
  // d0, d1 is relative coord within tile

  // z-order relative coord
  unsigned int serial = d0 + d1 * tile;
  z_order( z0, z1, serial );
  // add tile offset onto z-ordered index
  *z0 |= offset0;
  *z1 |= offset1;
}
#endif

/* global memory indices */
#define GLOBAL_C(IDX0I, IDX1J) (( (IDX0I)*strideC0I + (IDX1J)*strideC1J ))
#define GLOBAL_OFFSET_A(IDX0I, IDXK) (( (IDX0I)*strideA0I + (IDXK)*strideAK ))
#define GLOBAL_OFFSET_B(IDXK, IDX1J) (( (IDXK)*strideBK + (IDX1J)*strideB1J ))

/* data types */
#define DATA_TYPE half
#define MAC(A,B,DST) DST = __hfma(A,B,DST)

/* MAC's */
#define TYPE_MAC(MULA,MULB,DST) DST = MAC(MULA,MULB,DST);
#define TYPE_MAC_WRITE(DST,ALPHA,REG,BETA) DST = 0 != (BETA) ? (ALPHA)*(REG) + (BETA)*(DST) : (ALPHA)*(REG);

/* 4x4 micro-tile */
#define MAC_4x4\
  TYPE_MAC(rA[0],rB[0],rC[0+0*TT0I]); \
  TYPE_MAC(rA[1],rB[0],rC[1+0*TT0I]); \
  TYPE_MAC(rA[2],rB[0],rC[2+0*TT0I]); \
  TYPE_MAC(rA[3],rB[0],rC[3+0*TT0I]); \
  TYPE_MAC(rA[0],rB[1],rC[0+1*TT0I]); \
  TYPE_MAC(rA[1],rB[1],rC[1+1*TT0I]); \
  TYPE_MAC(rA[2],rB[1],rC[2+1*TT0I]); \
  TYPE_MAC(rA[3],rB[1],rC[3+1*TT0I]); \
  TYPE_MAC(rA[0],rB[2],rC[0+2*TT0I]); \
  TYPE_MAC(rA[1],rB[2],rC[1+2*TT0I]); \
  TYPE_MAC(rA[2],rB[2],rC[2+2*TT0I]); \
  TYPE_MAC(rA[3],rB[2],rC[3+2*TT0I]); \
  TYPE_MAC(rA[0],rB[3],rC[0+3*TT0I]); \
  TYPE_MAC(rA[1],rB[3],rC[1+3*TT0I]); \
  TYPE_MAC(rA[2],rB[3],rC[2+3*TT0I]); \
  TYPE_MAC(rA[3],rB[3],rC[3+3*TT0I]); \

/* hard-coded initial strides */
#define strideC0I 1
#define strideA0I 1
#define strideBK 1

  /******************************************/
  /* Begin Kernel                           */
  /******************************************/
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-parameter"
extern "C"
__global__ void Cij_Aik_Bkj_HB_MT032x032x08_K1(
  hipLaunchParm lp,
  half *C,
  half const * __restrict__ A,
  half const * __restrict__ B,
  half const alpha,
  half const beta,
  unsigned int const offsetC,
  unsigned int const offsetA,
  unsigned int const offsetB,
  unsigned int const strideC1J,
  unsigned int const strideAK,
  unsigned int const strideB1J,
  unsigned int const size0I,
  unsigned int const size1J,
  unsigned int const sizeK )
#pragma clang diagnostic pop
 {

  /******************************************/
  /* Allocate Resources                     */
  /******************************************/
#define SCALAR_ZERO (half)(0)

  /* registers for MAC's */
  DATA_TYPE rC[TT0I*TT1J];
  rC[0] = SCALAR_ZERO;
  rC[1] = SCALAR_ZERO;
  rC[2] = SCALAR_ZERO;
  rC[3] = SCALAR_ZERO;
  rC[4] = SCALAR_ZERO;
  rC[5] = SCALAR_ZERO;
  rC[6] = SCALAR_ZERO;
  rC[7] = SCALAR_ZERO;
  rC[8] = SCALAR_ZERO;
  rC[9] = SCALAR_ZERO;
  rC[10] = SCALAR_ZERO;
  rC[11] = SCALAR_ZERO;
  rC[12] = SCALAR_ZERO;
  rC[13] = SCALAR_ZERO;
  rC[14] = SCALAR_ZERO;
  rC[15] = SCALAR_ZERO;
  DATA_TYPE rA[TT0I];
  DATA_TYPE rB[TT1J];

  /* registers for global->local */
  DATA_TYPE a_0_0_0_0;
  DATA_TYPE a_0_1_0_0;
  DATA_TYPE a_0_2_0_0;
  DATA_TYPE a_0_3_0_0;
  DATA_TYPE b_0_0_0_0;
  DATA_TYPE b_0_1_0_0;
  DATA_TYPE b_0_2_0_0;
  DATA_TYPE b_0_3_0_0;

  /* allocate local memory */
  __shared__ DATA_TYPE localMemory[LDS_NUM_ELEMENTS];

  /******************************************/
  /* Global Read Addresses                  */
  /******************************************/

  /* global read addresses: subgroup */
  unsigned int serial = hc_get_workitem_id(0);
  unsigned int sgId = serial / (SG0I*SG1J);

  /* global read addresses: work-group */
  unsigned int wg0I = hc_get_group_id(0);
  unsigned int wg1J = hc_get_group_id(1);
  unsigned int nwg0I = hc_get_num_groups(0);
  unsigned int nwg1J = hc_get_num_groups(1);

  /* global read addresses: tile offset assignment a */
  unsigned int globalReadOffsetA0I = (serial%LVCA)*GLOBAL_LOAD_VECTOR_WIDTH_A + (wg0I)*MT0I;

  /* global read addresses: tile offset assignment b */
  unsigned int globalReadOffsetB1J = (serial/LVCB) + (wg1J)*MT1J;

  /* global read addresses: unroll assignment a */
  unsigned int globalReadOffsetAK = (serial/LVCA);

  /* global read addresses: unroll assignment b */
  unsigned int globalReadOffsetBK = (serial%LVCB)*GLOBAL_LOAD_VECTOR_WIDTH_B;

  /* global read addresses: tile offsets a */
  unsigned int globalReadOffsetA0I_0_0 = globalReadOffsetA0I + 0 + 0*LSCA;
  unsigned int globalReadOffsetA0I_0_1 = globalReadOffsetA0I + 1 + 0*LSCA;
  unsigned int globalReadOffsetA0I_0_2 = globalReadOffsetA0I + 2 + 0*LSCA;
  unsigned int globalReadOffsetA0I_0_3 = globalReadOffsetA0I + 3 + 0*LSCA;

  /* global read addresses: tile offsets b */
  unsigned int globalReadOffsetB1J_0_0 = globalReadOffsetB1J + 0 + 0*LSPB;

  /* global read addresses: unroll offsets a */
  unsigned int globalReadOffsetAK_0_0 = globalReadOffsetAK + 0 + 0*LSPA;
  unsigned int globalReadOffsetAK_0_1 = globalReadOffsetAK + 1 + 0*LSPA;
  unsigned int globalReadOffsetAK_0_2 = globalReadOffsetAK + 2 + 0*LSPA;
  unsigned int globalReadOffsetAK_0_3 = globalReadOffsetAK + 3 + 0*LSPA;

  /* global read addresses: unroll offsets b */
  unsigned int globalReadOffsetBK_0_0 = globalReadOffsetBK + 0 + 0*LSCB;
  unsigned int globalReadOffsetBK_0_1 = globalReadOffsetBK + 1 + 0*LSCB;
  unsigned int globalReadOffsetBK_0_2 = globalReadOffsetBK + 2 + 0*LSCB;
  unsigned int globalReadOffsetBK_0_3 = globalReadOffsetBK + 3 + 0*LSCB;

  /* global read addresses: shift a */
  globalReadOffsetA0I_0_0 = (  globalReadOffsetA0I_0_0 > size0I-GLOBAL_LOAD_VECTOR_WIDTH_A+0) ? size0I-GLOBAL_LOAD_VECTOR_WIDTH_A+0 : globalReadOffsetA0I_0_0;
  globalReadOffsetA0I_0_1 = (  globalReadOffsetA0I_0_1 > size0I-GLOBAL_LOAD_VECTOR_WIDTH_A+1) ? size0I-GLOBAL_LOAD_VECTOR_WIDTH_A+1 : globalReadOffsetA0I_0_1;
  globalReadOffsetA0I_0_2 = (  globalReadOffsetA0I_0_2 > size0I-GLOBAL_LOAD_VECTOR_WIDTH_A+2) ? size0I-GLOBAL_LOAD_VECTOR_WIDTH_A+2 : globalReadOffsetA0I_0_2;
  globalReadOffsetA0I_0_3 = (  globalReadOffsetA0I_0_3 > size0I-GLOBAL_LOAD_VECTOR_WIDTH_A+3) ? size0I-GLOBAL_LOAD_VECTOR_WIDTH_A+3 : globalReadOffsetA0I_0_3;

  /* global read addresses: shift b */
  globalReadOffsetB1J_0_0 = (  globalReadOffsetB1J_0_0 > size1J-1) ? size1J-1 : globalReadOffsetB1J_0_0;

  /* global read addresses: final offsets a */
  uint64_t globalReadOffsetA_0_0_0_0 = GLOBAL_OFFSET_A( globalReadOffsetA0I_0_0, globalReadOffsetAK_0_0 );
  uint64_t globalReadOffsetA_0_1_0_0 = GLOBAL_OFFSET_A( globalReadOffsetA0I_0_1, globalReadOffsetAK_0_0 );
  uint64_t globalReadOffsetA_0_2_0_0 = GLOBAL_OFFSET_A( globalReadOffsetA0I_0_2, globalReadOffsetAK_0_0 );
  uint64_t globalReadOffsetA_0_3_0_0 = GLOBAL_OFFSET_A( globalReadOffsetA0I_0_3, globalReadOffsetAK_0_0 );

  /* global read addresses: final offsets b */
  uint64_t globalReadOffsetB_0_0_0_0 = GLOBAL_OFFSET_B( globalReadOffsetBK_0_0, globalReadOffsetB1J_0_0 );
  uint64_t globalReadOffsetB_0_1_0_0 = GLOBAL_OFFSET_B( globalReadOffsetBK_0_1, globalReadOffsetB1J_0_0 );
  uint64_t globalReadOffsetB_0_2_0_0 = GLOBAL_OFFSET_B( globalReadOffsetBK_0_2, globalReadOffsetB1J_0_0 );
  uint64_t globalReadOffsetB_0_3_0_0 = GLOBAL_OFFSET_B( globalReadOffsetBK_0_3, globalReadOffsetB1J_0_0 );

  /* global read addresses: apply user offsets */
  C += offsetC;
  A += offsetA;
  B += offsetB;

  /* global read addresses: addresses a */
  DATA_TYPE const *globalReadA_0_0_0_0 = A + globalReadOffsetA_0_0_0_0;
  DATA_TYPE const *globalReadA_0_1_0_0 = A + globalReadOffsetA_0_1_0_0;
  DATA_TYPE const *globalReadA_0_2_0_0 = A + globalReadOffsetA_0_2_0_0;
  DATA_TYPE const *globalReadA_0_3_0_0 = A + globalReadOffsetA_0_3_0_0;

  /* global read addresses: addresses b */
  DATA_TYPE const *globalReadB_0_0_0_0 = B + globalReadOffsetB_0_0_0_0;
  DATA_TYPE const *globalReadB_0_1_0_0 = B + globalReadOffsetB_0_1_0_0;
  DATA_TYPE const *globalReadB_0_2_0_0 = B + globalReadOffsetB_0_2_0_0;
  DATA_TYPE const *globalReadB_0_3_0_0 = B + globalReadOffsetB_0_3_0_0;

  /* global read addresses: increments a */
  int64_t globalReadIncAK = (int64_t)strideAK*LOCAL_DEPTHU;

  /* global read addresses: increments b */
  int64_t globalReadIncBK = (int64_t)strideBK*LOCAL_DEPTHU;

  /******************************************/
  /* Local Write Addresses                  */
  /******************************************/

  /* local write addresses: tile assignment a */
  unsigned int lwA0I = (serial%LVCA)*GLOBAL_LOAD_VECTOR_WIDTH_A;

  /* local write addresses: tile assignment b */
  unsigned int lwB1J = (serial/LVCB);

  /* local write addresses: unroll assignment a */
  unsigned int lwAK = (serial/LVCA);

  /* local write addresses: unroll assignment b */
  unsigned int lwBK = (serial%LVCB)*GLOBAL_LOAD_VECTOR_WIDTH_B;

  /* local write addresses: first offset a */
  unsigned int localWriteFirstOffsetA = lwA0I + lwAK*(MT0I+PAD);

  /* local write addresses: first offset b */
  unsigned int localWriteFirstOffsetB = lwB1J + lwBK*(MT1J+PAD) + LDS_OFFSET_B;

  /* local write addresses: final offsets a */
  unsigned int localWriteOffsetA_0_0_0_0 = localWriteFirstOffsetA + (0 + 0*LSCA) + (0 + 0*LSPA)*(MT0I+PAD);
  unsigned int localWriteOffsetA_0_1_0_0 = localWriteFirstOffsetA + (1 + 0*LSCA) + (0 + 0*LSPA)*(MT0I+PAD);
  unsigned int localWriteOffsetA_0_2_0_0 = localWriteFirstOffsetA + (2 + 0*LSCA) + (0 + 0*LSPA)*(MT0I+PAD);
  unsigned int localWriteOffsetA_0_3_0_0 = localWriteFirstOffsetA + (3 + 0*LSCA) + (0 + 0*LSPA)*(MT0I+PAD);

  /* local write addresses: final offsets b */
  unsigned int localWriteOffsetB_0_0_0_0 = localWriteFirstOffsetB + (0 + 0*LSCB)*(MT1J+PAD) + (0 + 0*LSPB);
  unsigned int localWriteOffsetB_0_0_0_1 = localWriteFirstOffsetB + (1 + 0*LSCB)*(MT1J+PAD) + (0 + 0*LSPB);
  unsigned int localWriteOffsetB_0_0_0_2 = localWriteFirstOffsetB + (2 + 0*LSCB)*(MT1J+PAD) + (0 + 0*LSPB);
  unsigned int localWriteOffsetB_0_0_0_3 = localWriteFirstOffsetB + (3 + 0*LSCB)*(MT1J+PAD) + (0 + 0*LSPB);

  /* local write addresses: declare addresses a */
  DATA_TYPE *localWriteA_0_0_0_0;
  DATA_TYPE *localWriteA_0_1_0_0;
  DATA_TYPE *localWriteA_0_2_0_0;
  DATA_TYPE *localWriteA_0_3_0_0;

  /* local write addresses: declare addresses b */
  DATA_TYPE *localWriteB_0_0_0_0;
  DATA_TYPE *localWriteB_0_0_0_1;
  DATA_TYPE *localWriteB_0_0_0_2;
  DATA_TYPE *localWriteB_0_0_0_3;

  /* local write addresses: init pointers a */
  localWriteA_0_0_0_0 = (DATA_TYPE *)(localMemory + localWriteOffsetA_0_0_0_0);
  localWriteA_0_1_0_0 = (DATA_TYPE *)(localMemory + localWriteOffsetA_0_1_0_0);
  localWriteA_0_2_0_0 = (DATA_TYPE *)(localMemory + localWriteOffsetA_0_2_0_0);
  localWriteA_0_3_0_0 = (DATA_TYPE *)(localMemory + localWriteOffsetA_0_3_0_0);

  /* local write addresses: init pointers b */
  localWriteB_0_0_0_0 = (DATA_TYPE *)(localMemory + localWriteOffsetB_0_0_0_0);
  localWriteB_0_0_0_1 = (DATA_TYPE *)(localMemory + localWriteOffsetB_0_0_0_1);
  localWriteB_0_0_0_2 = (DATA_TYPE *)(localMemory + localWriteOffsetB_0_0_0_2);
  localWriteB_0_0_0_3 = (DATA_TYPE *)(localMemory + localWriteOffsetB_0_0_0_3);

  /******************************************/
  /* Local Read Addresses                   */
  /******************************************/

  /* local read addresses: tile assignments a */
  unsigned int lr0I = (serial % SG0I);

  /* local read addresses: tile assignments b */
  unsigned int lr1J = (serial / SG0I) % SG1J;

  /* local read addresses: final offsets a */
  unsigned int localReadOffsetA = lr0I*VECTOR_WIDTH + sgId*(MT0I+PAD);

  /* local read addresses: final offsets b */
  unsigned int localReadOffsetB = lr1J*VECTOR_WIDTH + sgId*(MT1J+PAD) + LDS_OFFSET_B;

  /* local read addresses: declare addresses a */
  DATA_TYPE *localReadA;

  /* local read addresses: declare addresses b */
  DATA_TYPE *localReadB;

  /* local read addresses: init pointers a */
  localReadA = (DATA_TYPE *)(localMemory + localReadOffsetA);

  /* local read addresses: init pointers b */
  localReadB = (DATA_TYPE *)(localMemory + localReadOffsetB);

  /* declare loop num iterations */
  unsigned int numIterK;
  numIterK = sizeK / LOCAL_DEPTHU;

  /******************************************/
  /* Unrolled Loop - Begin                  */
  /******************************************/
  while (numIterK-- > 0) {

    /* global read a */
    a_0_0_0_0 = *(globalReadA_0_0_0_0);
    a_0_1_0_0 = *(globalReadA_0_1_0_0);
    a_0_2_0_0 = *(globalReadA_0_2_0_0);
    a_0_3_0_0 = *(globalReadA_0_3_0_0);

    /* global read b */
    b_0_0_0_0 = *(globalReadB_0_0_0_0);
    b_0_1_0_0 = *(globalReadB_0_1_0_0);
    b_0_2_0_0 = *(globalReadB_0_2_0_0);
    b_0_3_0_0 = *(globalReadB_0_3_0_0);

    /* global read inc a */
    globalReadA_0_0_0_0 = (DATA_TYPE const *)( ((DATA_TYPE const *)globalReadA_0_0_0_0) + globalReadIncAK);
    globalReadA_0_1_0_0 = (DATA_TYPE const *)( ((DATA_TYPE const *)globalReadA_0_1_0_0) + globalReadIncAK);
    globalReadA_0_2_0_0 = (DATA_TYPE const *)( ((DATA_TYPE const *)globalReadA_0_2_0_0) + globalReadIncAK);
    globalReadA_0_3_0_0 = (DATA_TYPE const *)( ((DATA_TYPE const *)globalReadA_0_3_0_0) + globalReadIncAK);

    /* global read inc b */
    globalReadB_0_0_0_0 = (DATA_TYPE const *)( ((DATA_TYPE const *)globalReadB_0_0_0_0) + globalReadIncBK);
    globalReadB_0_1_0_0 = (DATA_TYPE const *)( ((DATA_TYPE const *)globalReadB_0_1_0_0) + globalReadIncBK);
    globalReadB_0_2_0_0 = (DATA_TYPE const *)( ((DATA_TYPE const *)globalReadB_0_2_0_0) + globalReadIncBK);
    globalReadB_0_3_0_0 = (DATA_TYPE const *)( ((DATA_TYPE const *)globalReadB_0_3_0_0) + globalReadIncBK);
    __syncthreads();

    /* local write a */
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wconditional-uninitialized"
    *localWriteA_0_0_0_0 = a_0_0_0_0;
    *localWriteA_0_1_0_0 = a_0_1_0_0;
    *localWriteA_0_2_0_0 = a_0_2_0_0;
    *localWriteA_0_3_0_0 = a_0_3_0_0;
#pragma clang diagnostic pop

    /* local write b */
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wconditional-uninitialized"
    *localWriteB_0_0_0_0 = b_0_0_0_0;
    *localWriteB_0_0_0_1 = b_0_1_0_0;
    *localWriteB_0_0_0_2 = b_0_2_0_0;
    *localWriteB_0_0_0_3 = b_0_3_0_0;
#pragma clang diagnostic pop
    __syncthreads();

    /* iter 0 */

    /* local read a */
    rA[0*VECTOR_WIDTH+0] = localReadA[0*SG0I*VECTOR_WIDTH + 0]; 
    rA[0*VECTOR_WIDTH+1] = localReadA[0*SG0I*VECTOR_WIDTH + 1]; 
    rA[0*VECTOR_WIDTH+2] = localReadA[0*SG0I*VECTOR_WIDTH + 2]; 
    rA[0*VECTOR_WIDTH+3] = localReadA[0*SG0I*VECTOR_WIDTH + 3]; 

    /* local read b */
    rB[0*VECTOR_WIDTH+0] = localReadB[0*SG1J*VECTOR_WIDTH + 0]; 
    rB[0*VECTOR_WIDTH+1] = localReadB[0*SG1J*VECTOR_WIDTH + 1]; 
    rB[0*VECTOR_WIDTH+2] = localReadB[0*SG1J*VECTOR_WIDTH + 2]; 
    rB[0*VECTOR_WIDTH+3] = localReadB[0*SG1J*VECTOR_WIDTH + 3]; 

    /* local read increment a */
    localReadA += LOCAL_SPLITU*(MT0I+PAD);

    /* local read increment b */
    localReadB += LOCAL_SPLITU*(MT1J+PAD);
    MAC_4x4

    /* iter 1 */

    /* local read a */
    rA[0*VECTOR_WIDTH+0] = localReadA[0*SG0I*VECTOR_WIDTH + 0]; 
    rA[0*VECTOR_WIDTH+1] = localReadA[0*SG0I*VECTOR_WIDTH + 1]; 
    rA[0*VECTOR_WIDTH+2] = localReadA[0*SG0I*VECTOR_WIDTH + 2]; 
    rA[0*VECTOR_WIDTH+3] = localReadA[0*SG0I*VECTOR_WIDTH + 3]; 

    /* local read b */
    rB[0*VECTOR_WIDTH+0] = localReadB[0*SG1J*VECTOR_WIDTH + 0]; 
    rB[0*VECTOR_WIDTH+1] = localReadB[0*SG1J*VECTOR_WIDTH + 1]; 
    rB[0*VECTOR_WIDTH+2] = localReadB[0*SG1J*VECTOR_WIDTH + 2]; 
    rB[0*VECTOR_WIDTH+3] = localReadB[0*SG1J*VECTOR_WIDTH + 3]; 

    /* local read increment a */
    localReadA += LOCAL_SPLITU*(MT0I+PAD);

    /* local read increment b */
    localReadB += LOCAL_SPLITU*(MT1J+PAD);
    MAC_4x4

    /* iter 2 */

    /* local read a */
    rA[0*VECTOR_WIDTH+0] = localReadA[0*SG0I*VECTOR_WIDTH + 0]; 
    rA[0*VECTOR_WIDTH+1] = localReadA[0*SG0I*VECTOR_WIDTH + 1]; 
    rA[0*VECTOR_WIDTH+2] = localReadA[0*SG0I*VECTOR_WIDTH + 2]; 
    rA[0*VECTOR_WIDTH+3] = localReadA[0*SG0I*VECTOR_WIDTH + 3]; 

    /* local read b */
    rB[0*VECTOR_WIDTH+0] = localReadB[0*SG1J*VECTOR_WIDTH + 0]; 
    rB[0*VECTOR_WIDTH+1] = localReadB[0*SG1J*VECTOR_WIDTH + 1]; 
    rB[0*VECTOR_WIDTH+2] = localReadB[0*SG1J*VECTOR_WIDTH + 2]; 
    rB[0*VECTOR_WIDTH+3] = localReadB[0*SG1J*VECTOR_WIDTH + 3]; 

    /* local read increment a */
    localReadA += LOCAL_SPLITU*(MT0I+PAD);

    /* local read increment b */
    localReadB += LOCAL_SPLITU*(MT1J+PAD);
    MAC_4x4

    /* iter 3 */

    /* local read a */
    rA[0*VECTOR_WIDTH+0] = localReadA[0*SG0I*VECTOR_WIDTH + 0]; 
    rA[0*VECTOR_WIDTH+1] = localReadA[0*SG0I*VECTOR_WIDTH + 1]; 
    rA[0*VECTOR_WIDTH+2] = localReadA[0*SG0I*VECTOR_WIDTH + 2]; 
    rA[0*VECTOR_WIDTH+3] = localReadA[0*SG0I*VECTOR_WIDTH + 3]; 

    /* local read b */
    rB[0*VECTOR_WIDTH+0] = localReadB[0*SG1J*VECTOR_WIDTH + 0]; 
    rB[0*VECTOR_WIDTH+1] = localReadB[0*SG1J*VECTOR_WIDTH + 1]; 
    rB[0*VECTOR_WIDTH+2] = localReadB[0*SG1J*VECTOR_WIDTH + 2]; 
    rB[0*VECTOR_WIDTH+3] = localReadB[0*SG1J*VECTOR_WIDTH + 3]; 

    /* local read increment a */
    localReadA += LOCAL_SPLITU*(MT0I+PAD);

    /* local read increment b */
    localReadB += LOCAL_SPLITU*(MT1J+PAD);
    MAC_4x4

    /* iter 4 */

    /* local read a */
    rA[0*VECTOR_WIDTH+0] = localReadA[0*SG0I*VECTOR_WIDTH + 0]; 
    rA[0*VECTOR_WIDTH+1] = localReadA[0*SG0I*VECTOR_WIDTH + 1]; 
    rA[0*VECTOR_WIDTH+2] = localReadA[0*SG0I*VECTOR_WIDTH + 2]; 
    rA[0*VECTOR_WIDTH+3] = localReadA[0*SG0I*VECTOR_WIDTH + 3]; 

    /* local read b */
    rB[0*VECTOR_WIDTH+0] = localReadB[0*SG1J*VECTOR_WIDTH + 0]; 
    rB[0*VECTOR_WIDTH+1] = localReadB[0*SG1J*VECTOR_WIDTH + 1]; 
    rB[0*VECTOR_WIDTH+2] = localReadB[0*SG1J*VECTOR_WIDTH + 2]; 
    rB[0*VECTOR_WIDTH+3] = localReadB[0*SG1J*VECTOR_WIDTH + 3]; 

    /* local read increment a */
    localReadA += LOCAL_SPLITU*(MT0I+PAD);

    /* local read increment b */
    localReadB += LOCAL_SPLITU*(MT1J+PAD);
    MAC_4x4

    /* iter 5 */

    /* local read a */
    rA[0*VECTOR_WIDTH+0] = localReadA[0*SG0I*VECTOR_WIDTH + 0]; 
    rA[0*VECTOR_WIDTH+1] = localReadA[0*SG0I*VECTOR_WIDTH + 1]; 
    rA[0*VECTOR_WIDTH+2] = localReadA[0*SG0I*VECTOR_WIDTH + 2]; 
    rA[0*VECTOR_WIDTH+3] = localReadA[0*SG0I*VECTOR_WIDTH + 3]; 

    /* local read b */
    rB[0*VECTOR_WIDTH+0] = localReadB[0*SG1J*VECTOR_WIDTH + 0]; 
    rB[0*VECTOR_WIDTH+1] = localReadB[0*SG1J*VECTOR_WIDTH + 1]; 
    rB[0*VECTOR_WIDTH+2] = localReadB[0*SG1J*VECTOR_WIDTH + 2]; 
    rB[0*VECTOR_WIDTH+3] = localReadB[0*SG1J*VECTOR_WIDTH + 3]; 

    /* local read increment a */
    localReadA += LOCAL_SPLITU*(MT0I+PAD);

    /* local read increment b */
    localReadB += LOCAL_SPLITU*(MT1J+PAD);
    MAC_4x4

    /* iter 6 */

    /* local read a */
    rA[0*VECTOR_WIDTH+0] = localReadA[0*SG0I*VECTOR_WIDTH + 0]; 
    rA[0*VECTOR_WIDTH+1] = localReadA[0*SG0I*VECTOR_WIDTH + 1]; 
    rA[0*VECTOR_WIDTH+2] = localReadA[0*SG0I*VECTOR_WIDTH + 2]; 
    rA[0*VECTOR_WIDTH+3] = localReadA[0*SG0I*VECTOR_WIDTH + 3]; 

    /* local read b */
    rB[0*VECTOR_WIDTH+0] = localReadB[0*SG1J*VECTOR_WIDTH + 0]; 
    rB[0*VECTOR_WIDTH+1] = localReadB[0*SG1J*VECTOR_WIDTH + 1]; 
    rB[0*VECTOR_WIDTH+2] = localReadB[0*SG1J*VECTOR_WIDTH + 2]; 
    rB[0*VECTOR_WIDTH+3] = localReadB[0*SG1J*VECTOR_WIDTH + 3]; 

    /* local read inc a */
    localReadA += LOCAL_SPLITU*(MT0I+PAD);

    /* local read inc b */
    localReadB += LOCAL_SPLITU*(MT1J+PAD);
    MAC_4x4

    /* iter 7 */

    /* local read a */
    rA[0*VECTOR_WIDTH+0] = localReadA[0*SG0I*VECTOR_WIDTH + 0]; 
    rA[0*VECTOR_WIDTH+1] = localReadA[0*SG0I*VECTOR_WIDTH + 1]; 
    rA[0*VECTOR_WIDTH+2] = localReadA[0*SG0I*VECTOR_WIDTH + 2]; 
    rA[0*VECTOR_WIDTH+3] = localReadA[0*SG0I*VECTOR_WIDTH + 3]; 

    /* local read b */
    rB[0*VECTOR_WIDTH+0] = localReadB[0*SG1J*VECTOR_WIDTH + 0]; 
    rB[0*VECTOR_WIDTH+1] = localReadB[0*SG1J*VECTOR_WIDTH + 1]; 
    rB[0*VECTOR_WIDTH+2] = localReadB[0*SG1J*VECTOR_WIDTH + 2]; 
    rB[0*VECTOR_WIDTH+3] = localReadB[0*SG1J*VECTOR_WIDTH + 3]; 

    /* local read init pointers a */
    localReadA = (DATA_TYPE *)(localMemory + localReadOffsetA);

    /* local read init pointers b */
    localReadB = (DATA_TYPE *)(localMemory + localReadOffsetB);
    MAC_4x4

    /******************************************/
    /* Unrolled Loop - End                    */
    /******************************************/
  }

  /* shift vector components d0 */
  unsigned int wgMT0I = size0I - wg0I*MT0I;
  if (wgMT0I > MT0I) wgMT0I = MT0I;
  unsigned int r0I = wgMT0I % VECTOR_WIDTH;
  if (r0I > 0 && ((wgMT0I/VECTOR_WIDTH)%SG0I) == serial % SG0I ) {
    unsigned int s0I = (wgMT0I/VECTOR_WIDTH)/SG0I;
    if (r0I == 1) {
      {
        rC[0+0*VECTOR_WIDTH+0*TT0I] = rC[3+0*VECTOR_WIDTH+0*TT0I];
        rC[0+0*VECTOR_WIDTH+1*TT0I] = rC[3+0*VECTOR_WIDTH+1*TT0I];
        rC[0+0*VECTOR_WIDTH+2*TT0I] = rC[3+0*VECTOR_WIDTH+2*TT0I];
        rC[0+0*VECTOR_WIDTH+3*TT0I] = rC[3+0*VECTOR_WIDTH+3*TT0I];
      }
    }
    if (r0I == 2) {
      {
        rC[0+0*VECTOR_WIDTH+0*TT0I] = rC[2+0*VECTOR_WIDTH+0*TT0I];
        rC[1+0*VECTOR_WIDTH+0*TT0I] = rC[3+0*VECTOR_WIDTH+0*TT0I];
        rC[0+0*VECTOR_WIDTH+1*TT0I] = rC[2+0*VECTOR_WIDTH+1*TT0I];
        rC[1+0*VECTOR_WIDTH+1*TT0I] = rC[3+0*VECTOR_WIDTH+1*TT0I];
        rC[0+0*VECTOR_WIDTH+2*TT0I] = rC[2+0*VECTOR_WIDTH+2*TT0I];
        rC[1+0*VECTOR_WIDTH+2*TT0I] = rC[3+0*VECTOR_WIDTH+2*TT0I];
        rC[0+0*VECTOR_WIDTH+3*TT0I] = rC[2+0*VECTOR_WIDTH+3*TT0I];
        rC[1+0*VECTOR_WIDTH+3*TT0I] = rC[3+0*VECTOR_WIDTH+3*TT0I];
      }
    }
    if (r0I == 3) {
      {
        rC[0+0*VECTOR_WIDTH+0*TT0I] = rC[1+0*VECTOR_WIDTH+0*TT0I];
        rC[1+0*VECTOR_WIDTH+0*TT0I] = rC[2+0*VECTOR_WIDTH+0*TT0I];
        rC[2+0*VECTOR_WIDTH+0*TT0I] = rC[3+0*VECTOR_WIDTH+0*TT0I];
        rC[0+0*VECTOR_WIDTH+1*TT0I] = rC[1+0*VECTOR_WIDTH+1*TT0I];
        rC[1+0*VECTOR_WIDTH+1*TT0I] = rC[2+0*VECTOR_WIDTH+1*TT0I];
        rC[2+0*VECTOR_WIDTH+1*TT0I] = rC[3+0*VECTOR_WIDTH+1*TT0I];
        rC[0+0*VECTOR_WIDTH+2*TT0I] = rC[1+0*VECTOR_WIDTH+2*TT0I];
        rC[1+0*VECTOR_WIDTH+2*TT0I] = rC[2+0*VECTOR_WIDTH+2*TT0I];
        rC[2+0*VECTOR_WIDTH+2*TT0I] = rC[3+0*VECTOR_WIDTH+2*TT0I];
        rC[0+0*VECTOR_WIDTH+3*TT0I] = rC[1+0*VECTOR_WIDTH+3*TT0I];
        rC[1+0*VECTOR_WIDTH+3*TT0I] = rC[2+0*VECTOR_WIDTH+3*TT0I];
        rC[2+0*VECTOR_WIDTH+3*TT0I] = rC[3+0*VECTOR_WIDTH+3*TT0I];
      }
    }
  }

  /* not-LocalSplitU: global write indices */
  unsigned int globalC0I = (wg0I)*MT0I + (serial % SG0I)*VECTOR_WIDTH;
  unsigned int globalC1J = (wg1J)*MT1J + (serial / SG0I)*VECTOR_WIDTH;

  /* not-LocalSplitU: global write */
  if (globalC0I + 0 + 0*SG0I*VECTOR_WIDTH < size0I) {  if (globalC1J + 0 + 0*SG1J*VECTOR_WIDTH < size1J) {  TYPE_MAC_WRITE( C[ GLOBAL_C( (uint64_t) globalC0I + 0 + 0*SG0I*VECTOR_WIDTH, (uint64_t) globalC1J + 0 + 0*SG1J*VECTOR_WIDTH) ], alpha, rC[0*VECTOR_WIDTH+0 + (0*VECTOR_WIDTH+0)*TT0I], beta) } }
  if (globalC0I + 1 + 0*SG0I*VECTOR_WIDTH < size0I) {  if (globalC1J + 0 + 0*SG1J*VECTOR_WIDTH < size1J) {  TYPE_MAC_WRITE( C[ GLOBAL_C( (uint64_t) globalC0I + 1 + 0*SG0I*VECTOR_WIDTH, (uint64_t) globalC1J + 0 + 0*SG1J*VECTOR_WIDTH) ], alpha, rC[0*VECTOR_WIDTH+1 + (0*VECTOR_WIDTH+0)*TT0I], beta) } }
  if (globalC0I + 2 + 0*SG0I*VECTOR_WIDTH < size0I) {  if (globalC1J + 0 + 0*SG1J*VECTOR_WIDTH < size1J) {  TYPE_MAC_WRITE( C[ GLOBAL_C( (uint64_t) globalC0I + 2 + 0*SG0I*VECTOR_WIDTH, (uint64_t) globalC1J + 0 + 0*SG1J*VECTOR_WIDTH) ], alpha, rC[0*VECTOR_WIDTH+2 + (0*VECTOR_WIDTH+0)*TT0I], beta) } }
  if (globalC0I + 3 + 0*SG0I*VECTOR_WIDTH < size0I) {  if (globalC1J + 0 + 0*SG1J*VECTOR_WIDTH < size1J) {  TYPE_MAC_WRITE( C[ GLOBAL_C( (uint64_t) globalC0I + 3 + 0*SG0I*VECTOR_WIDTH, (uint64_t) globalC1J + 0 + 0*SG1J*VECTOR_WIDTH) ], alpha, rC[0*VECTOR_WIDTH+3 + (0*VECTOR_WIDTH+0)*TT0I], beta) } }
  if (globalC0I + 0 + 0*SG0I*VECTOR_WIDTH < size0I) {  if (globalC1J + 1 + 0*SG1J*VECTOR_WIDTH < size1J) {  TYPE_MAC_WRITE( C[ GLOBAL_C( (uint64_t) globalC0I + 0 + 0*SG0I*VECTOR_WIDTH, (uint64_t) globalC1J + 1 + 0*SG1J*VECTOR_WIDTH) ], alpha, rC[0*VECTOR_WIDTH+0 + (0*VECTOR_WIDTH+1)*TT0I], beta) } }
  if (globalC0I + 1 + 0*SG0I*VECTOR_WIDTH < size0I) {  if (globalC1J + 1 + 0*SG1J*VECTOR_WIDTH < size1J) {  TYPE_MAC_WRITE( C[ GLOBAL_C( (uint64_t) globalC0I + 1 + 0*SG0I*VECTOR_WIDTH, (uint64_t) globalC1J + 1 + 0*SG1J*VECTOR_WIDTH) ], alpha, rC[0*VECTOR_WIDTH+1 + (0*VECTOR_WIDTH+1)*TT0I], beta) } }
  if (globalC0I + 2 + 0*SG0I*VECTOR_WIDTH < size0I) {  if (globalC1J + 1 + 0*SG1J*VECTOR_WIDTH < size1J) {  TYPE_MAC_WRITE( C[ GLOBAL_C( (uint64_t) globalC0I + 2 + 0*SG0I*VECTOR_WIDTH, (uint64_t) globalC1J + 1 + 0*SG1J*VECTOR_WIDTH) ], alpha, rC[0*VECTOR_WIDTH+2 + (0*VECTOR_WIDTH+1)*TT0I], beta) } }
  if (globalC0I + 3 + 0*SG0I*VECTOR_WIDTH < size0I) {  if (globalC1J + 1 + 0*SG1J*VECTOR_WIDTH < size1J) {  TYPE_MAC_WRITE( C[ GLOBAL_C( (uint64_t) globalC0I + 3 + 0*SG0I*VECTOR_WIDTH, (uint64_t) globalC1J + 1 + 0*SG1J*VECTOR_WIDTH) ], alpha, rC[0*VECTOR_WIDTH+3 + (0*VECTOR_WIDTH+1)*TT0I], beta) } }
  if (globalC0I + 0 + 0*SG0I*VECTOR_WIDTH < size0I) {  if (globalC1J + 2 + 0*SG1J*VECTOR_WIDTH < size1J) {  TYPE_MAC_WRITE( C[ GLOBAL_C( (uint64_t) globalC0I + 0 + 0*SG0I*VECTOR_WIDTH, (uint64_t) globalC1J + 2 + 0*SG1J*VECTOR_WIDTH) ], alpha, rC[0*VECTOR_WIDTH+0 + (0*VECTOR_WIDTH+2)*TT0I], beta) } }
  if (globalC0I + 1 + 0*SG0I*VECTOR_WIDTH < size0I) {  if (globalC1J + 2 + 0*SG1J*VECTOR_WIDTH < size1J) {  TYPE_MAC_WRITE( C[ GLOBAL_C( (uint64_t) globalC0I + 1 + 0*SG0I*VECTOR_WIDTH, (uint64_t) globalC1J + 2 + 0*SG1J*VECTOR_WIDTH) ], alpha, rC[0*VECTOR_WIDTH+1 + (0*VECTOR_WIDTH+2)*TT0I], beta) } }
  if (globalC0I + 2 + 0*SG0I*VECTOR_WIDTH < size0I) {  if (globalC1J + 2 + 0*SG1J*VECTOR_WIDTH < size1J) {  TYPE_MAC_WRITE( C[ GLOBAL_C( (uint64_t) globalC0I + 2 + 0*SG0I*VECTOR_WIDTH, (uint64_t) globalC1J + 2 + 0*SG1J*VECTOR_WIDTH) ], alpha, rC[0*VECTOR_WIDTH+2 + (0*VECTOR_WIDTH+2)*TT0I], beta) } }
  if (globalC0I + 3 + 0*SG0I*VECTOR_WIDTH < size0I) {  if (globalC1J + 2 + 0*SG1J*VECTOR_WIDTH < size1J) {  TYPE_MAC_WRITE( C[ GLOBAL_C( (uint64_t) globalC0I + 3 + 0*SG0I*VECTOR_WIDTH, (uint64_t) globalC1J + 2 + 0*SG1J*VECTOR_WIDTH) ], alpha, rC[0*VECTOR_WIDTH+3 + (0*VECTOR_WIDTH+2)*TT0I], beta) } }
  if (globalC0I + 0 + 0*SG0I*VECTOR_WIDTH < size0I) {  if (globalC1J + 3 + 0*SG1J*VECTOR_WIDTH < size1J) {  TYPE_MAC_WRITE( C[ GLOBAL_C( (uint64_t) globalC0I + 0 + 0*SG0I*VECTOR_WIDTH, (uint64_t) globalC1J + 3 + 0*SG1J*VECTOR_WIDTH) ], alpha, rC[0*VECTOR_WIDTH+0 + (0*VECTOR_WIDTH+3)*TT0I], beta) } }
  if (globalC0I + 1 + 0*SG0I*VECTOR_WIDTH < size0I) {  if (globalC1J + 3 + 0*SG1J*VECTOR_WIDTH < size1J) {  TYPE_MAC_WRITE( C[ GLOBAL_C( (uint64_t) globalC0I + 1 + 0*SG0I*VECTOR_WIDTH, (uint64_t) globalC1J + 3 + 0*SG1J*VECTOR_WIDTH) ], alpha, rC[0*VECTOR_WIDTH+1 + (0*VECTOR_WIDTH+3)*TT0I], beta) } }
  if (globalC0I + 2 + 0*SG0I*VECTOR_WIDTH < size0I) {  if (globalC1J + 3 + 0*SG1J*VECTOR_WIDTH < size1J) {  TYPE_MAC_WRITE( C[ GLOBAL_C( (uint64_t) globalC0I + 2 + 0*SG0I*VECTOR_WIDTH, (uint64_t) globalC1J + 3 + 0*SG1J*VECTOR_WIDTH) ], alpha, rC[0*VECTOR_WIDTH+2 + (0*VECTOR_WIDTH+3)*TT0I], beta) } }
  if (globalC0I + 3 + 0*SG0I*VECTOR_WIDTH < size0I) {  if (globalC1J + 3 + 0*SG1J*VECTOR_WIDTH < size1J) {  TYPE_MAC_WRITE( C[ GLOBAL_C( (uint64_t) globalC0I + 3 + 0*SG0I*VECTOR_WIDTH, (uint64_t) globalC1J + 3 + 0*SG1J*VECTOR_WIDTH) ], alpha, rC[0*VECTOR_WIDTH+3 + (0*VECTOR_WIDTH+3)*TT0I], beta) } }

}
#undef UNROLL
#undef LOCAL_SPLITU
#undef LOCAL_DEPTHU
#undef SG0I
#undef SG1J
#undef TT0I
#undef TT1J
#undef MT0I
#undef MT1J
#undef NLCA
#undef NLCB
#undef NLPA
#undef NLPB
#undef LSCA
#undef LSPA
#undef LSCB
#undef LSPB
#undef GLOBAL_C
#undef GLOBAL_OFFSET_A
#undef GLOBAL_OFFSET_B
#undef DATA_TYPE
#undef LDS_OFFSET_B
#undef LDS_OFFSET_BLK
#undef LDS_NUM_ELEMENTS
#undef NUM_THREADS
#undef WORK_GROUP_MAPPING
#undef VECTOR_WIDTH
#undef GLOBAL_LOAD_VECTOR_WIDTH_A
#undef GLOBAL_LOAD_VECTOR_WIDTH_B
#undef TYPE_MAC
#undef TYPE_MAC_WRITE
#undef GLOBAL_SPLITU
#undef VECTOR_ZERO
#undef SCALAR_ZERO
#undef MAC_4x4
#undef strideC0I
#undef strideA0I
#undef strideBK



