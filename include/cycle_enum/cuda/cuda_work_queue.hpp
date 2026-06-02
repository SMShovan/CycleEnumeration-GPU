#pragma once

#include "cycle_enum/core/graph_view.hpp"
#include "cycle_enum/core/histogram.hpp"

#include <cstddef>

/**
 * @file cuda_work_queue.hpp
 * @brief Persistent-block work-queue scheduling for CUDA cycle counting.
 */

namespace cycle_enum::cuda {

/**
 * @brief A persistent-kernel launch configuration.
 */
struct WorkQueueLaunch {
  unsigned int block_size = 0; ///< Threads per block.
  unsigned int grid_blocks = 0; ///< Resident blocks to launch.
};

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
 * @return Histogram keyed by cycle length.
 *
 * @throws std::invalid_argument if `max_cycle_length` is less than 2.
 * @throws std::runtime_error if CUDA support is unavailable in this build.
 * @throws std::out_of_range if the requested CUDA device is not visible.
 */
[[nodiscard]] CycleHistogram count_simple_cycles_johnson_work_queue(
    const GraphView& graph,
    int device_id,
    std::size_t max_cycle_length);

}  // namespace cycle_enum::cuda
