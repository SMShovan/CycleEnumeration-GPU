#pragma once

#include "cycle_enum/core/histogram.hpp"
#include "cycle_enum/dynamic/directed_graph.hpp"
#include "cycle_enum/dynamic/edge_change.hpp"

#include <cstddef>

/**
 * @file update_sequential.hpp
 * @brief Sequential incremental update of a static simple-cycle histogram.
 */

namespace cycle_enum::dynamic {

/**
 * @brief Incrementally update a static simple-cycle histogram under a batch.
 *
 * Given the initial graph, its prior histogram, and a batch of edge changes,
 * return the histogram of the post-batch graph without recomputing the
 * unchanged cycles. The batch is applied as delete-then-insert with edge-id
 * ownership: the delete phase enumerates cycles through each deleted edge in the
 * initial graph and subtracts them, then the insert phase enumerates cycles
 * through each inserted edge in the post-batch graph and adds them. Each affected
 * cycle is attributed to the smallest-id changed edge it contains.
 *
 * Cycles longer than `max_cycle_length` are not touched, so for the result to
 * equal a full recomputation the prior histogram must have been computed with
 * the same length cap.
 *
 * @param initial_graph Directed graph before the batch.
 * @param initial_histogram Prior histogram of `initial_graph`.
 * @param batch Edge changes to apply (normalized internally).
 * @param max_cycle_length Inclusive maximum cycle length; must be at least 2.
 * @return Histogram of the post-batch graph.
 *
 * @throws std::invalid_argument if `max_cycle_length` is less than 2.
 * @throws std::logic_error if the update would drive any bucket negative, which
 * indicates an inconsistent prior histogram.
 */
[[nodiscard]] CycleHistogram update_static_histogram(
    const DirectedGraph& initial_graph,
    const CycleHistogram& initial_histogram,
    const EdgeBatch& batch,
    std::size_t max_cycle_length);

}  // namespace cycle_enum::dynamic
