#pragma once

#include <cuda.h>
#include "cuda_runtime.h"

int GetCurrentDeviceId();

static int GetCudaDeviceCount();

cudaDeviceProp* GetDeviceProperties(int id);
