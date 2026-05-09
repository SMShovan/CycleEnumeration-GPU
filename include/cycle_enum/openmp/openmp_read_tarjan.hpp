#pragma once

#include "cycle_enum/core/graph_view.hpp"
#include "cycle_enum/core/histogram.hpp"

#include <optional>

/**
 * @file openmp_read_tarjan.hpp
 * @brief OpenMP Read-Tarjan-style static directed-cycle counter.
 */

namespace cycle_enum::openmp {

/**
 * @brief Count static simple cycles with coarse-grained OpenMP root work.
 *
 * The implementation keeps path and visited state local to each worker and
 * merges thread-local histograms after the parallel region. It preserves the
 * same smallest-root duplicate-avoidance rule as the sequential Read-Tarjan
 * baseline.
 */
[[nodiscard]] CycleHistogram count_simple_cycles_read_tarjan(
    const GraphView& graph,
    int requested_threads,
    std::optional<std::size_t> max_cycle_length = std::nullopt);

}  // namespace cycle_enum::openmp
