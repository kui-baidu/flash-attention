#include "cuda_utils.h"

#include <vector>
#include <thread>
#include <mutex>
#include <memory>
#include <algorithm>

static std::once_flag g_device_props_size_init_flag;
static std::vector<std::unique_ptr<std::once_flag>> g_device_props_init_flags;
static std::vector<cudaDeviceProp> g_device_props;

int GetCurrentDeviceId() {
  int device_id;
  cudaGetDevice(&device_id);
  return device_id;
}

static int GetCudaDeviceCount() {
  int driverVersion = 0;
  cudaError_t status = cudaDriverGetVersion(&driverVersion);

  if (!(status == cudaSuccess && driverVersion != 0)) {
    return 0;
  }

  const auto *cuda_visible_devices = std::getenv("CUDA_VISIBLE_DEVICES");

  if (cuda_visible_devices != nullptr) {
    std::string cuda_visible_devices_str(cuda_visible_devices);
    if (!cuda_visible_devices_str.empty()) {
      cuda_visible_devices_str.erase(
          0, cuda_visible_devices_str.find_first_not_of('\''));
      cuda_visible_devices_str.erase(
          cuda_visible_devices_str.find_last_not_of('\'') + 1);
      cuda_visible_devices_str.erase(
          0, cuda_visible_devices_str.find_first_not_of('\"'));
      cuda_visible_devices_str.erase(
          cuda_visible_devices_str.find_last_not_of('\"') + 1);
    }
    if (std::all_of(cuda_visible_devices_str.begin(),
                    cuda_visible_devices_str.end(),
                    [](char ch) { return ch == ' '; })) {
      return 0;
    }
  }
  int count;
  cudaGetDeviceCount(&count);
  return count;
}

cudaDeviceProp* GetDeviceProperties(int id) {
  std::call_once(g_device_props_size_init_flag, [&] {
    int gpu_num = 0;
    gpu_num = GetCudaDeviceCount();
    g_device_props_init_flags.resize(gpu_num);
    g_device_props.resize(gpu_num);
    for (int i = 0; i < gpu_num; ++i) {
      g_device_props_init_flags[i] = std::make_unique<std::once_flag>();
    }
  });

  if (id == -1) {
    id = GetCurrentDeviceId();
  }

  std::call_once(*(g_device_props_init_flags[id]), [&] {
        cudaGetDeviceProperties(&g_device_props[id], id);
  });

  return &g_device_props[id];
}

