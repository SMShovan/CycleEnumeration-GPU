#pragma once

#include "cycle_enum/core/histogram.hpp"
#include "cycle_enum/dynamic/directed_graph.hpp"
#include "cycle_enum/dynamic/edge_change.hpp"

#include <cstddef>

/**
 * @file update_cuda.hpp
 * @brief Single-GPU incremental update of a static simple-cycle histogram.
 */

namespace cycle_enum::dynamic {

/**
 * @brief CUDA counterpart of `update_static_histogram`.
 *
 * Both phases launch one device work item per changed edge. Each work item
 * enumerates the cycles it owns through an explicit device depth-first search
 * with edge-id ownership pruning, accumulating per-length counts with atomics.
 * The delete phase runs on the initial graph and the insert phase on the
 * post-batch graph; the host forms the signed delta from the two add-only count
 * buffers and applies it to the prior histogram. The result is identical to the
 * sequential update.
 *
 * @param initial_graph Directed graph before the batch.
 * @param initial_histogram Prior histogram of `initial_graph`.
 * @param batch Edge changes to apply (normalized internally).
 * @param max_cycle_length Inclusive maximum cycle length; must be at least 2.
 * @param device_id CUDA device ordinal.
 * @return Histogram of the post-batch graph.
 *
 * @throws std::invalid_argument if `max_cycle_length` is less than 2.
 * @throws std::runtime_error if CUDA support is unavailable in this build.
 * @throws std::out_of_range if the requested CUDA device is not visible.
 */
[[nodiscard]] CycleHistogram update_static_histogram_cuda(
    const DirectedGraph& initial_graph,
    const CycleHistogram& initial_histogram,
    const EdgeBatch& batch,
    std::size_t max_cycle_length,
    int device_id);

}  // namespace cycle_enum::dynamic
