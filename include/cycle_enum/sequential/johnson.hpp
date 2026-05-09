#pragma once

#include "cycle_enum/core/graph_view.hpp"
#include "cycle_enum/core/histogram.hpp"

#include <optional>

/**
 * @file johnson.hpp
 * @brief Sequential Johnson-style directed simple-cycle counter.
 */

namespace cycle_enum::sequential {

/**
 * @brief Count static directed simple cycles with Johnson blocked-set pruning.
 *
 * The implementation counts each directed cycle once by treating the smallest
 * compact vertex in a cycle as that cycle's root. It is a correctness-first
 * sequential baseline; SCC decomposition and other performance refinements can
 * be added later without changing this public interface.
 *
 * @param graph CSR/CSC graph view.
 * @param max_cycle_length Optional maximum cycle length to explore.
 * @return Histogram keyed by cycle length.
 *
 * @throws std::invalid_argument if `max_cycle_length` is less than 2.
 */
[[nodiscard]] CycleHistogram count_simple_cycles_johnson(
    const GraphView& graph,
    std::optional<std::size_t> max_cycle_length = std::nullopt);

/**
 * @brief Count simple cycles under a start-edge time-window constraint.
 *
 * The timestamp boundary rule follows the baseline implementation so results
 * can be compared with the TBB code on temporal graph inputs.
 */
[[nodiscard]] CycleHistogram count_time_window_cycles_johnson(
    const GraphView& graph,
    Timestamp window_width,
    std::optional<std::size_t> max_cycle_length = std::nullopt);

}  // namespace cycle_enum::sequential
