

  /* atomic add float */
#ifndef ATOMIC_FLOAT_FUNCTION
#define ATOMIC_FLOAT_FUNCTION

__device__ inline void atomicAddType(float *fPtr, float operand) 
{
  std::atomic<float> *aPtr = reinterpret_cast<std::atomic<float>*>(fPtr);
  float oldValue, newValue;
  oldValue = aPtr->load(std::memory_order_relaxed);
  do 
  {
    newValue = oldValue + operand;
  } 
  while ( !std::atomic_compare_exchange_weak_explicit(aPtr, &oldValue, newValue, std::memory_order_relaxed, std::memory_order_release) );
}
#endif

