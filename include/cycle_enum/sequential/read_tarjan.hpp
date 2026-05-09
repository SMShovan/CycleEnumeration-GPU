#pragma once

#include "cycle_enum/core/graph_view.hpp"
#include "cycle_enum/core/histogram.hpp"

#include <optional>

/**
 * @file read_tarjan.hpp
 * @brief Sequential Read-Tarjan-style directed simple-cycle counter.
 */

namespace cycle_enum::sequential {

/**
 * @brief Count static directed simple cycles using path-extension search.
 *
 * This implementation follows the Read-Tarjan idea of finding path extensions
 * back to the root and recursively branching when alternate extensions exist.
 * It uses the smallest compact vertex id as the cycle root to count each
 * directed cycle once.
 *
 * @param graph CSR/CSC graph view.
 * @param max_cycle_length Optional maximum cycle length to count.
 * @return Histogram keyed by cycle length.
 *
 * @throws std::invalid_argument if `max_cycle_length` is less than 2.
 */
[[nodiscard]] CycleHistogram count_simple_cycles_read_tarjan(
    const GraphView& graph,
    std::optional<std::size_t> max_cycle_length = std::nullopt);

/**
 * @brief Count simple cycles under a start-edge time-window constraint.
 *
 * This variant follows the same timestamp boundary convention as the
 * baseline-compatible Johnson time-window implementation.
 */
[[nodiscard]] CycleHistogram count_time_window_cycles_read_tarjan(
    const GraphView& graph,
    Timestamp window_width,
    std::optional<std::size_t> max_cycle_length = std::nullopt);

}  // namespace cycle_enum::sequential
