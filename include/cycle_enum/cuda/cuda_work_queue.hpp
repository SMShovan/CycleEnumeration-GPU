#pragma once

#include "cycle_enum/core/graph_view.hpp"
#include "cycle_enum/core/histogram.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

/**
 * @file cuda_work_queue.hpp
 * @brief Persistent-block work-queue scheduling for CUDA cycle counting.
 */

namespace cycle_enum::cuda {

/**
 * @brief Device-side timing breakdown for a single CUDA counting run.
 *
 * All values are milliseconds measured with CUDA events on the device. They are
 * only populated by entry points that accept a `CudaTimingsMs*`; a null pointer
 * disables timing. `memcpy_ms` covers the host-to-device graph upload plus the
 * device-to-host histogram copy; `kernel_ms` is the counting kernel alone;
 * `total_ms` spans the whole device-side routine (upload, allocation, kernel,
 * and download).
 */
struct CudaTimingsMs {
  float kernel_ms = 0.0F; ///< Counting-kernel execution time.
  float memcpy_ms = 0.0F; ///< Host/device transfer time (upload + download).
  float total_ms = 0.0F;  ///< End-to-end device-side time.
};

/**
 * @brief Per-root certainty report for the blocking backend.
 *
 * The blocking counter never overcounts, so its histogram is a lower bound. A
 * root is flagged when its bounded search hit the length cap on an otherwise
 * eligible descent (or overflowed a per-thread capacity), meaning its cycles may
 * be undercounted. Unflagged roots are exact, so `flagged_count == 0` certifies
 * the histogram equals the exact (non-blocking) result.
 */
struct UncertaintyReport {
  std::uint64_t flagged_count = 0; ///< Roots that may be undercounted.
  std::uint64_t total_roots = 0;   ///< Roots processed.
  std::vector<VertexId> flagged;   ///< Ids of flagged roots (optional detail).
};

/**
 * @brief A persistent-kernel launch configuration.
 */
struct WorkQueueLaunch {
  unsigned int block_size = 0; ///< Threads per block.
  unsigned int grid_blocks = 0; ///< Resident blocks to launch.
};

/**
 * @brief Tunable persistent-kernel launch parameters.
 */
struct WorkQueueTuning {
  unsigned int block_size = 128; ///< Threads per block.
  unsigned int blocks_per_sm = 16; ///< Resident blocks per multiprocessor.
};

/**
 * @brief Read work-queue tuning parameters from the environment.
 *
 * `CYCLE_ENUM_CUDA_BLOCK_SIZE` and `CYCLE_ENUM_CUDA_BLOCKS_PER_SM` override the
 * defaults so a tuning sweep can vary the launch without rebuilding. An unset
 * variable keeps the default; a present but non-positive or unparseable value is
 * rejected so sweep typos fail loudly. The block size must be a positive
 * multiple of the 32-thread warp size.
 *
 * @throws std::invalid_argument if a present value is invalid.
 */
[[nodiscard]] WorkQueueTuning work_queue_tuning_from_env();

/**
 * @brief Plan a persistent-kernel launch for a work-queue kernel.
 *
 * A persistent kernel launches a fixed wave of blocks that stay resident and
 * pull work items from a global atomic counter, instead of mapping one work
 * item to one thread. This attacks the load imbalance of the naive kernels,
 * where a few heavy work items stall whole warps while most threads idle.
 *
 * The planned grid fills the device with `blocks_per_sm * sm_count` resident
 * blocks, but never launches more blocks than there are work items, since extra
 * blocks would claim nothing. The result is empty when there is no work.
 *
 * @param work_item_count Number of independent work items (for example roots).
 * @param block_size Threads per block; must be positive.
 * @param blocks_per_sm Resident blocks per streaming multiprocessor; positive.
 * @param sm_count Streaming multiprocessor count of the target device; positive.
 * @return The block size and resident grid size to launch.
 *
 * @throws std::invalid_argument if any of the configuration values is zero.
 */
[[nodiscard]] WorkQueueLaunch plan_work_queue_launch(
    std::size_t work_item_count,
    unsigned int block_size,
    unsigned int blocks_per_sm,
    unsigned int sm_count);

/**
 * @brief Count static simple cycles with a persistent work-queue kernel.
 *
 * This is the dynamically scheduled counterpart to
 * `count_simple_cycles_johnson`. Persistent blocks claim root vertices from a
 * global counter so that work is balanced across the device regardless of how
 * skewed the per-root search trees are. The result is identical to the naive
 * static counter.
 *
 * @param graph CSR/CSC graph view.
 * @param device_id CUDA device ordinal.
 * @param max_cycle_length Maximum cycle length to count; must be at least 2.
 * @param timings Optional output for the device-side timing breakdown; when
 *   non-null it is filled with kernel, transfer, and total milliseconds.
 * @return Histogram keyed by cycle length.
 *
 * @throws std::invalid_argument if `max_cycle_length` is less than 2.
 * @throws std::runtime_error if CUDA support is unavailable in this build.
 * @throws std::out_of_range if the requested CUDA device is not visible.
 */
[[nodiscard]] CycleHistogram count_simple_cycles_johnson_work_queue(
    const GraphView& graph,
    int device_id,
    std::size_t max_cycle_length,
    CudaTimingsMs* timings = nullptr,
    bool blocking = false,
    UncertaintyReport* report = nullptr);

/**
 * @brief Count static simple cycles with the rank-based depth-aware blocking
 * backend (the proposed method).
 *
 * Exact for all cycles up to `max_cycle_length` when `step_budget == 0`. When
 * `step_budget > 0`, any root whose search exceeds the budget is stopped and
 * flagged, so the histogram becomes a certified lower bound and `report` (if
 * provided) lists the flagged roots.
 */
[[nodiscard]] CycleHistogram count_simple_cycles_johnson_rank(
    const GraphView& graph,
    int device_id,
    std::size_t max_cycle_length,
    CudaTimingsMs* timings = nullptr,
    std::uint64_t step_budget = 0,
    UncertaintyReport* report = nullptr);

}  // namespace cycle_enum::cuda
