#include "utils.h"

void SetZero(void *ptr, size_t sizeof_type, std::initializer_list<int> shapes, cudaStream_t stream) {
    size_t n = sizeof_type;
    for (int s : shapes) n *= s;
    cudaMemsetAsync(ptr, 0, n, stream);
}

template <typename T>
static __global__ void FillConstantKernel(T *ptr, T value, size_t n) {
  auto idx = static_cast<size_t>(blockDim.x) * blockIdx.x + threadIdx.x;
  if (idx < n) {
    ptr[idx] = value;
  }
} 

template <typename T>
void SetConstValue(void *ptr, T value, size_t n, cudaStream_t stream) {
  constexpr auto kNumThreads = 1024;
  auto block = (n + kNumThreads - 1) / kNumThreads; 
  FillConstantKernel<T><<<block, kNumThreads, 0, stream>>>(static_cast<T *>(ptr), value, n);
} 

template
void SetConstValue(void *ptr, float value, size_t n, cudaStream_t stream);
