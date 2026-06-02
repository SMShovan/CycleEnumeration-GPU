#pragma once

#include "cycle_enum/core/graph_view.hpp"
#include "cycle_enum/core/histogram.hpp"

#include <cstddef>
#include <optional>

/**
 * @file openmp_read_tarjan_tasks.hpp
 * @brief Fine-grained OpenMP task experiment for Read-Tarjan cycle counting.
 */

namespace cycle_enum::openmp {

/**
 * @brief Count static simple cycles with fine-grained OpenMP tasks.
 *
 * This is a measurable experiment, not the primary CPU baseline. Unlike the
 * coarse-grained Read-Tarjan counter, which assigns one root vertex to one
 * loop iteration, this version spawns an OpenMP task for every path prefix up
 * to `task_cutoff_depth` and then finishes each subtree serially. Each spawned
 * task carries a private copy of the path and visited state, mirroring the
 * copy-on-steal idea from the baseline paper.
 *
 * A larger `task_cutoff_depth` produces finer-grained tasks and can improve
 * load balance on skewed graphs, at the cost of more task-creation overhead and
 * more state copying. `task_cutoff_depth` must be at least 1, where 1 reproduces
 * one task per root.
 *
 * The result is identical to the sequential and coarse-grained Read-Tarjan
 * counters; only the scheduling differs. Without an OpenMP runtime, or for a
 * single-thread request, the function runs the same serial search.
 *
 * @param graph CSR/CSC graph view.
 * @param requested_threads Requested OpenMP thread count.
 * @param max_cycle_length Optional inclusive maximum cycle length.
 * @param task_cutoff_depth Path length below which child branches are spawned
 * as tasks; must be at least 1.
 * @return Histogram keyed by cycle length.
 *
 * @throws std::invalid_argument if `max_cycle_length` is less than 2 or
 * `task_cutoff_depth` is 0.
 * @throws std::runtime_error if more than one thread is requested without an
 * OpenMP runtime.
 */
[[nodiscard]] CycleHistogram count_simple_cycles_read_tarjan_tasks(
    const GraphView& graph,
    int requested_threads,
    std::optional<std::size_t> max_cycle_length = std::nullopt,
    std::size_t task_cutoff_depth = 2);

}  // namespace cycle_enum::openmp
