#pragma once

#include "cycle_enum/core/graph_view.hpp"
#include "cycle_enum/core/histogram.hpp"

#include <cstdint>
#include <optional>

/**
 * @file openmp_temporal_read_tarjan.hpp
 * @brief OpenMP temporal Read-Tarjan-style directed-cycle counter.
 */

namespace cycle_enum::openmp {

/**
 * @brief Search counters collected by the OpenMP temporal Read-Tarjan baseline.
 */
struct TemporalReadTarjanStats {
  std::uint64_t dfs_states = 0; ///< Number of timestamped DFS states entered.
  std::uint64_t closing_time_prunes = 0; ///< States skipped by closing times.
  std::uint64_t timestamp_extensions = 0; ///< Timestamp transitions explored.
};

/**
 * @brief Histogram plus merged thread-local temporal Read-Tarjan counters.
 */
struct TemporalReadTarjanResult {
  CycleHistogram histogram; ///< Counted temporal cycles by length.
  TemporalReadTarjanStats stats; ///< Search instrumentation.
};

/**
 * @brief Count temporal simple cycles with OpenMP root-level work sharing.
 *
 * The implementation preserves the sequential temporal Read-Tarjan semantics:
 * every valid timestamp assignment contributes to the histogram, edge
 * timestamps must be strictly increasing, and each timestamped start edge owns
 * its time window.
 *
 * @param graph CSR/CSC graph view.
 * @param window_width Non-negative time-window width in timestamp units.
 * @param requested_threads Requested OpenMP thread count.
 * @param max_cycle_length Optional maximum cycle length to count.
 * @return Histogram keyed by cycle length.
 *
 * @throws std::invalid_argument if `window_width` is negative,
 * `requested_threads` is not positive, or `max_cycle_length` is less than 2.
 * @throws std::runtime_error if more than one thread is requested in a build
 * without OpenMP support.
 * @throws std::overflow_error if a start timestamp plus window width overflows.
 */
[[nodiscard]] CycleHistogram count_temporal_cycles_read_tarjan(
    const GraphView& graph,
    Timestamp window_width,
    int requested_threads,
    std::optional<std::size_t> max_cycle_length = std::nullopt);

/**
 * @brief Count temporal cycles and return merged OpenMP search counters.
 */
[[nodiscard]] TemporalReadTarjanResult
count_temporal_cycles_read_tarjan_with_stats(
    const GraphView& graph,
    Timestamp window_width,
    int requested_threads,
    std::optional<std::size_t> max_cycle_length = std::nullopt);

}  // namespace cycle_enum::openmp
