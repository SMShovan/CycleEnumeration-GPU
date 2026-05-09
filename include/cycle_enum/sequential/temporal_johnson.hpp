#pragma once

#include "cycle_enum/core/graph_view.hpp"
#include "cycle_enum/core/histogram.hpp"

#include <cstdint>
#include <optional>

/**
 * @file temporal_johnson.hpp
 * @brief Sequential temporal directed-cycle counter.
 */

namespace cycle_enum::sequential {

/**
 * @brief Search counters for the sequential temporal Johnson implementation.
 *
 * These counters are intended for tests, tuning notes, and benchmark reports.
 * They are not part of the cycle-counting result itself.
 */
struct TemporalJohnsonStats {
  std::uint64_t dfs_states = 0; ///< Number of timestamped DFS states entered.
  std::uint64_t closing_time_prunes = 0; ///< States skipped by closing times.
  std::uint64_t bundled_arrival_timestamps = 0; ///< Arrival timestamps bundled.
};

/**
 * @brief Histogram plus optional instrumentation for temporal Johnson runs.
 */
struct TemporalJohnsonResult {
  CycleHistogram histogram; ///< Counted temporal cycles by length.
  TemporalJohnsonStats stats; ///< Search instrumentation.
};

/**
 * @brief Count temporal simple cycles using a correctness-first DFS.
 *
 * A temporal cycle starts from one timestamped edge event. Every later edge
 * event must be strictly later than the previous event and no later than the
 * start timestamp plus `window_width`. This baseline deliberately keeps the
 * state explicit and local so closing-time and bundling optimizations can be
 * added without changing the public result contract.
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
[[nodiscard]] CycleHistogram count_temporal_cycles_johnson(
    const GraphView& graph,
    Timestamp window_width,
    std::optional<std::size_t> max_cycle_length = std::nullopt);

/**
 * @brief Count temporal cycles and return search instrumentation.
 *
 * The histogram is identical to `count_temporal_cycles_johnson`; the additional
 * counters make pruning effects visible in tests and benchmark reports.
 */
[[nodiscard]] TemporalJohnsonResult count_temporal_cycles_johnson_with_stats(
    const GraphView& graph,
    Timestamp window_width,
    std::optional<std::size_t> max_cycle_length = std::nullopt);

}  // namespace cycle_enum::sequential
