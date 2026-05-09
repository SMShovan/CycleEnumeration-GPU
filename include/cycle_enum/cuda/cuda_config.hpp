#pragma once

#include <cstddef>
#include <string>
#include <vector>

/**
 * @file cuda_config.hpp
 * @brief Host-visible CUDA runtime availability and device queries.
 */

namespace cycle_enum::cuda {

/**
 * @brief Summary of one CUDA-capable device visible to the runtime.
 */
struct CudaDeviceInfo {
  int device_id = 0; ///< CUDA runtime device ordinal.
  std::string name; ///< Human-readable device name from cudaDeviceProp.
  int compute_capability_major = 0; ///< Major compute capability.
  int compute_capability_minor = 0; ///< Minor compute capability.
  int multiprocessor_count = 0; ///< Number of streaming multiprocessors.
  std::size_t global_memory_bytes = 0; ///< Total global memory in bytes.
};

/**
 * @brief Return true when this build was compiled with CUDA runtime support.
 */
[[nodiscard]] bool compiled_with_cuda() noexcept;

/**
 * @brief Return the number of CUDA devices visible to the runtime.
 *
 * Non-CUDA builds return zero. CUDA builds throw `std::runtime_error` if the
 * runtime query fails for a reason other than no visible device.
 */
[[nodiscard]] int device_count();

/**
 * @brief Enumerate CUDA devices visible to this process.
 */
[[nodiscard]] std::vector<CudaDeviceInfo> enumerate_devices();

/**
 * @brief Return a short human-readable CUDA availability summary.
 */
[[nodiscard]] std::string availability_string();

/**
 * @brief Validate that a CUDA device id can be used by later kernels.
 *
 * @throws std::invalid_argument if `device_id` is negative.
 * @throws std::runtime_error if CUDA support is not compiled in.
 * @throws std::out_of_range if the requested device id is not visible.
 */
void require_device(int device_id);

}  // namespace cycle_enum::cuda
