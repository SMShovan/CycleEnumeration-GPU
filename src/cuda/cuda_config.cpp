#include "cycle_enum/cuda/cuda_config.hpp"

#include <sstream>
#include <stdexcept>

/**
 * @file cuda_config.cpp
 * @brief CUDA runtime detection helpers with non-CUDA build fallbacks.
 */

/**
 * @def CYCLE_ENUM_CUDA_ENABLED
 * @brief Compile-time flag set by CMake when CUDA runtime support is linked.
 */
#ifndef CYCLE_ENUM_CUDA_ENABLED
#define CYCLE_ENUM_CUDA_ENABLED 0
#endif

#if CYCLE_ENUM_CUDA_ENABLED
#include <cuda_runtime_api.h>
#endif

namespace cycle_enum::cuda {

namespace {

#if CYCLE_ENUM_CUDA_ENABLED
std::runtime_error cuda_runtime_error(const char* operation,
                                      const cudaError_t error) {
  std::ostringstream message;
  message << operation << " failed: " << cudaGetErrorString(error);
  return std::runtime_error(message.str());
}
#endif

}  // namespace

bool compiled_with_cuda() noexcept {
#if CYCLE_ENUM_CUDA_ENABLED
  return true;
#else
  return false;
#endif
}

int device_count() {
#if CYCLE_ENUM_CUDA_ENABLED
  int count = 0;
  const cudaError_t error = cudaGetDeviceCount(&count);
  if (error == cudaErrorNoDevice) {
    return 0;
  }
  if (error != cudaSuccess) {
    throw cuda_runtime_error("cudaGetDeviceCount", error);
  }
  return count;
#else
  return 0;
#endif
}

std::vector<CudaDeviceInfo> enumerate_devices() {
  std::vector<CudaDeviceInfo> devices;
#if CYCLE_ENUM_CUDA_ENABLED
  const int count = device_count();
  devices.reserve(static_cast<std::size_t>(count));

  for (int device_id = 0; device_id < count; ++device_id) {
    cudaDeviceProp properties{};
    const cudaError_t error = cudaGetDeviceProperties(&properties, device_id);
    if (error != cudaSuccess) {
      throw cuda_runtime_error("cudaGetDeviceProperties", error);
    }

    devices.push_back(CudaDeviceInfo{
        device_id,
        properties.name,
        properties.major,
        properties.minor,
        properties.multiProcessorCount,
        static_cast<std::size_t>(properties.totalGlobalMem),
    });
  }
#endif
  return devices;
}

std::string availability_string() {
  if (!compiled_with_cuda()) {
    return "CUDA support is not compiled into this build";
  }

#if CYCLE_ENUM_CUDA_ENABLED
  try {
    const int count = device_count();
    if (count == 0) {
      return "CUDA support is compiled, but no CUDA device is visible";
    }

    std::ostringstream message;
    message << "CUDA support is compiled; visible devices: " << count;
    return message.str();
  } catch (const std::exception& error) {
    std::ostringstream message;
    message << "CUDA runtime query failed: " << error.what();
    return message.str();
  }
#else
  return "CUDA support is not compiled into this build";
#endif
}

void require_device(const int device_id) {
  if (device_id < 0) {
    throw std::invalid_argument("cuda device id must be non-negative");
  }
  if (!compiled_with_cuda()) {
    throw std::runtime_error("CUDA support is not compiled into this build");
  }

  const int count = device_count();
  if (device_id >= count) {
    std::ostringstream message;
    message << "CUDA device " << device_id << " is not visible";
    throw std::out_of_range(message.str());
  }
}

}  // namespace cycle_enum::cuda
