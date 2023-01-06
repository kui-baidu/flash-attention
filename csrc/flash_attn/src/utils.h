#include <cuda.h>
#include <vector>
#include <initializer_list>

void SetZero(void *ptr, size_t sizeof_type, std::initializer_list<int> shapes, cudaStream_t stream);

template <typename T>
void SetConstValue(void *ptr, T value, size_t n, cudaStream_t stream);

