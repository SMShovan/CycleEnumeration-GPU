#pragma once

#include "cycle_enum/core/graph_view.hpp"
#include "cycle_enum/core/histogram.hpp"

#include <optional>

/**
 * @file temporal_read_tarjan.hpp
 * @brief Sequential temporal Read-Tarjan-style cycle counter.
 */

namespace cycle_enum::sequential {

/**
 * @brief Count temporal simple cycles with Read-Tarjan-style local state.
 *
 * The implementation uses independent start edge events and local DFS state,
 * matching the work-item shape needed for later OpenMP and CUDA versions. Edge
 * timestamps must be strictly increasing and remain within the start edge's
 * time window.
 *
 * @param graph CSR/CSC graph view.
 * @param window_width Non-negative time-window width in timestamp units.
 * @param max_cycle_length Optional maximum cycle length to count.
 * @return Histogram keyed by cycle length.
 *
 * @throws std::invalid_argument if `window_width` is negative or
 * `max_cycle_length` is less than 2.
 * @throws std::overflow_error if a start timestamp plus window width overflows.
 */
[[nodiscard]] CycleHistogram count_temporal_cycles_read_tarjan(
    const GraphView& graph,
    Timestamp window_width,
    std::optional<std::size_t> max_cycle_length = std::nullopt);

}  // namespace cycle_enum::sequential
