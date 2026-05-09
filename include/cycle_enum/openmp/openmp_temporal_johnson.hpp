#pragma once

#include "cycle_enum/core/graph_view.hpp"
#include "cycle_enum/core/histogram.hpp"

#include <cstdint>
#include <optional>

/**
 * @file openmp_temporal_johnson.hpp
 * @brief OpenMP temporal directed-cycle counter with Johnson-style DFS state.
 */

namespace cycle_enum::openmp {

/**
 * @brief Search counters collected by the OpenMP temporal Johnson baseline.
 *
 * The counters mirror the sequential temporal Johnson instrumentation so that
 * pruning and timestamp-bundling behavior can be compared across CPU backends.
 */
struct TemporalJohnsonStats {
  std::uint64_t dfs_states = 0; ///< Number of timestamped DFS states entered.
  std::uint64_t closing_time_prunes = 0; ///< States skipped by closing times.
  std::uint64_t bundled_arrival_timestamps = 0; ///< Arrival timestamps bundled.
};

/**
 * @brief Histogram plus merged thread-local instrumentation.
 */
struct TemporalJohnsonResult {
  CycleHistogram histogram; ///< Counted temporal cycles by length.
  TemporalJohnsonStats stats; ///< Search instrumentation.
};

/**
 * @brief Count temporal simple cycles with OpenMP root-level parallelism.
 *
 * Each worker owns its DFS stack, visited flags, closing-time cache, and local
 * histogram. The final result is produced by merging thread-local histograms
 * and counters, which avoids synchronization in the recursive search.
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
[[nodiscard]] CycleHistogram count_temporal_cycles_johnson(
    const GraphView& graph,
    Timestamp window_width,
    int requested_threads,
    std::optional<std::size_t> max_cycle_length = std::nullopt);

/**
 * @brief Count temporal cycles and return merged search instrumentation.
 *
 * The histogram is identical to `count_temporal_cycles_johnson`; the counters
 * make pruning and path-bundling effects visible in tests and benchmark notes.
 */
[[nodiscard]] TemporalJohnsonResult count_temporal_cycles_johnson_with_stats(
    const GraphView& graph,
    Timestamp window_width,
    int requested_threads,
    std::optional<std::size_t> max_cycle_length = std::nullopt);

}  // namespace cycle_enum::openmp
