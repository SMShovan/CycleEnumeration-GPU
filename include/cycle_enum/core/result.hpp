#pragma once

#include <cstdint>
#include <string>
#include <string_view>

/**
 * @file result.hpp
 * @brief Shared status and runtime reporting types.
 */

namespace cycle_enum {

/**
 * @brief High-level outcome of an algorithm run.
 */
enum class RunStatus {
  Success,        ///< The algorithm completed and produced a result.
  InvalidOptions, ///< The requested options failed validation.
  Unsupported,    ///< The selected backend or mode is not implemented.
  RuntimeError,   ///< Execution failed after validation succeeded.
  Skipped         ///< The run was intentionally skipped, usually in tests.
};

/**
 * @brief Runtime counters and timing breakdown for a cycle-counting run.
 *
 * Timings use seconds to keep CLI, benchmark, and report output consistent.
 * CUDA-specific timings remain zero for CPU-only runs.
 */
struct RuntimeStatistics {
  double graph_load_seconds = 0.0;      ///< Input parsing and graph build time.
  double preprocessing_seconds = 0.0;   ///< Optional pruning or setup time.
  double cpu_seconds = 0.0;             ///< Host algorithm time.
  double transfer_h2d_seconds = 0.0;    ///< Host-to-device transfer time.
  double kernel_seconds = 0.0;          ///< CUDA kernel execution time.
  double transfer_d2h_seconds = 0.0;    ///< Device-to-host transfer time.
  double total_seconds = 0.0;           ///< End-to-end measured runtime.
  std::uint64_t vertex_visits = 0;      ///< Optional search-work counter.
  std::uint64_t edge_visits = 0;        ///< Optional search-work counter.
  std::uint64_t start_work_items = 0;   ///< Number of independent starts.
};

/**
 * @brief Lightweight status object returned by high-level drivers.
 */
struct RunResult {
  RunStatus status = RunStatus::Success; ///< Run outcome.
  std::string message;                   ///< Human-readable status detail.
  RuntimeStatistics statistics;          ///< Timing and work counters.
};

/**
 * @brief Convert a run status to a stable CLI/reporting string.
 */
[[nodiscard]] std::string_view to_string(RunStatus status) noexcept;

}  // namespace cycle_enum

