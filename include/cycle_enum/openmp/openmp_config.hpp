#pragma once

#include <string>

/**
 * @file openmp_config.hpp
 * @brief Runtime helpers for optional OpenMP support.
 */

namespace cycle_enum::openmp {

/**
 * @brief Summary of the OpenMP support compiled into the current binary.
 */
struct OpenMPConfig {
  bool available = false; ///< True when the binary was linked with OpenMP.
  int max_threads = 1; ///< Maximum OpenMP threads reported by the runtime.
};

/**
 * @brief Return the OpenMP configuration compiled into the binary.
 */
[[nodiscard]] OpenMPConfig config() noexcept;

/**
 * @brief Return true when OpenMP was enabled in CMake and found by the compiler.
 */
[[nodiscard]] bool available() noexcept;

/**
 * @brief Return the maximum OpenMP thread count, or 1 without OpenMP support.
 */
[[nodiscard]] int max_threads() noexcept;

/**
 * @brief Validate and resolve a requested OpenMP thread count.
 *
 * Without OpenMP support, a request for one thread is accepted so sequential
 * fallback paths can share option handling. Requests above one thread require a
 * real OpenMP runtime.
 *
 * @throws std::invalid_argument if `requested_threads` is not positive.
 * @throws std::runtime_error if more than one thread is requested without
 * OpenMP support.
 */
[[nodiscard]] int resolve_thread_count(int requested_threads);

/**
 * @brief Return a human-readable availability string for CLI and reports.
 */
[[nodiscard]] std::string availability_string();

}  // namespace cycle_enum::openmp
