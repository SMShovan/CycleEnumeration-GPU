#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

/**
 * @file options.hpp
 * @brief Shared configuration types for cycle enumeration backends.
 */

namespace cycle_enum {

/**
 * @brief Logical cycle-counting mode requested by the caller.
 */
enum class CycleMode {
  Simple,           ///< Static directed simple cycles.
  SimpleTimeWindow, ///< Simple cycles whose edges fall inside a start window.
  Temporal          ///< Temporal cycles with increasing edge timestamps.
};

/**
 * @brief Algorithm family used for a counting run.
 */
enum class AlgorithmFamily {
  Johnson,    ///< Johnson blocked-set algorithm.
  ReadTarjan, ///< Read-Tarjan path-extension algorithm.
  BruteForce  ///< Tiny-graph oracle used for tests and validation.
};

/**
 * @brief Hardware or runtime backend selected for execution.
 */
enum class ExecutionPolicy {
  Sequential, ///< Single-threaded CPU execution.
  OpenMP,     ///< OpenMP CPU execution.
  Cuda        ///< Single-GPU CUDA execution.
};

/**
 * @brief User-facing options shared by all implementations.
 *
 * The fields are intentionally backend-neutral. CUDA-specific launch tuning and
 * OpenMP-specific scheduling controls will live in narrower types later, while
 * this structure captures the behavior that affects correctness and reporting.
 */
struct CycleEnumerationOptions {
  AlgorithmFamily algorithm = AlgorithmFamily::Johnson; ///< Algorithm family.
  ExecutionPolicy execution = ExecutionPolicy::Sequential; ///< Runtime backend.
  CycleMode mode = CycleMode::Simple; ///< Cycle-counting semantics.

  /**
   * @brief Optional time-window width in timestamp units.
   *
   * `SimpleTimeWindow` and `Temporal` modes require a positive value. Static
   * `Simple` mode ignores this field.
   */
  std::optional<std::int64_t> time_window;

  /**
   * @brief Optional maximum cycle length.
   *
   * This is primarily a guardrail for experimental CUDA kernels that may need a
   * bounded device-side DFS stack. Exact CPU modes can leave it unset.
   */
  std::optional<std::size_t> max_cycle_length;

  int openmp_threads = 1; ///< Requested OpenMP thread count.
  int cuda_device_id = 0; ///< Requested CUDA device ordinal.
  bool use_cycle_union = false; ///< Enable temporal cycle-union pruning.
  bool validate_results = false; ///< Compare with a trusted slower backend.
};

/**
 * @brief Return the default options used by library callers and the CLI.
 */
[[nodiscard]] CycleEnumerationOptions default_options() noexcept;

/**
 * @brief Return true when a cycle mode requires a positive time window.
 */
[[nodiscard]] bool requires_time_window(CycleMode mode) noexcept;

/**
 * @brief Convert a cycle mode to a stable CLI/reporting string.
 */
[[nodiscard]] std::string_view to_string(CycleMode mode) noexcept;

/**
 * @brief Convert an algorithm family to a stable CLI/reporting string.
 */
[[nodiscard]] std::string_view to_string(AlgorithmFamily algorithm) noexcept;

/**
 * @brief Convert an execution policy to a stable CLI/reporting string.
 */
[[nodiscard]] std::string_view to_string(ExecutionPolicy execution) noexcept;

/**
 * @brief Validate shared options and return human-readable error messages.
 *
 * An empty vector means the options are structurally valid. This function does
 * not inspect the machine, so CUDA availability and OpenMP availability are
 * checked by backend-specific setup code.
 */
[[nodiscard]] std::vector<std::string> validate_options(
    const CycleEnumerationOptions& options);

}  // namespace cycle_enum

