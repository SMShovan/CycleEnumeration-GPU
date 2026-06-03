#pragma once

#include "cycle_enum/core/histogram.hpp"
#include "cycle_enum/dynamic/directed_graph.hpp"
#include "cycle_enum/dynamic/edge_change.hpp"

#include <cstddef>

/**
 * @file update_openmp.hpp
 * @brief OpenMP-parallel incremental update of a static simple-cycle histogram.
 */

namespace cycle_enum::dynamic {

/**
 * @brief OpenMP-parallel counterpart of `update_static_histogram`.
 *
 * The delete and insert phases each parallelize over their changed edges. Within
 * a phase the graph snapshot is read only, and edge-id ownership makes each
 * affected cycle the responsibility of exactly one edge, so the per-edge work
 * needs no coordination beyond a final reduction of the per-thread deltas. The
 * result is identical to the sequential update. Without an OpenMP runtime, or
 * for a single-thread request, it runs the sequential update.
 *
 * @param initial_graph Directed graph before the batch.
 * @param initial_histogram Prior histogram of `initial_graph`.
 * @param batch Edge changes to apply (normalized internally).
 * @param max_cycle_length Inclusive maximum cycle length; must be at least 2.
 * @param requested_threads Requested OpenMP thread count.
 * @return Histogram of the post-batch graph.
 *
 * @throws std::invalid_argument if `max_cycle_length` is less than 2.
 * @throws std::runtime_error if more than one thread is requested without an
 * OpenMP runtime.
 */
[[nodiscard]] CycleHistogram update_static_histogram_openmp(
    const DirectedGraph& initial_graph,
    const CycleHistogram& initial_histogram,
    const EdgeBatch& batch,
    std::size_t max_cycle_length,
    int requested_threads);

}  // namespace cycle_enum::dynamic
