#pragma once

#include <cuda.h>

int GetCurrentDeviceId();

static int GetCudaDeviceCount();

cudaDeviceProp* GetDeviceProperties(int id);
