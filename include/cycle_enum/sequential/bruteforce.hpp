#pragma once

#include "cycle_enum/core/graph_view.hpp"
#include "cycle_enum/core/histogram.hpp"

#include <optional>

/**
 * @file bruteforce.hpp
 * @brief Tiny-graph brute-force cycle-counting oracle.
 */

namespace cycle_enum::sequential {

/**
 * @brief Count static directed simple cycles using exhaustive DFS.
 *
 * This implementation is intentionally simple and is intended for tests and
 * validation, not production-scale graph runs. It counts each directed cycle
 * once by only allowing a cycle's root to be its smallest compact vertex id.
 *
 * @param graph CSR/CSC graph view.
 * @param max_cycle_length Optional maximum cycle length to explore.
 * @return Histogram keyed by cycle length.
 *
 * @throws std::invalid_argument if `max_cycle_length` is less than 2.
 */
[[nodiscard]] CycleHistogram count_simple_cycles_bruteforce(
    const GraphView& graph,
    std::optional<std::size_t> max_cycle_length = std::nullopt);

/**
 * @brief Count simple cycles whose edges fall in a start-edge time window.
 *
 * This oracle mirrors the baseline time-window duplicate-avoidance rule: edge
 * timestamp lookup is inclusive before the root vertex in numeric order and
 * strict after the root. It is intended for small validation fixtures.
 */
[[nodiscard]] CycleHistogram count_time_window_cycles_bruteforce(
    const GraphView& graph,
    Timestamp window_width,
    std::optional<std::size_t> max_cycle_length = std::nullopt);

}  // namespace cycle_enum::sequential
