#pragma once

#include "cycle_enum/core/graph_view.hpp"
#include "cycle_enum/core/histogram.hpp"

#include <optional>

/**
 * @file openmp_johnson.hpp
 * @brief OpenMP static directed-cycle counter.
 */

namespace cycle_enum::openmp {

/**
 * @brief Count static simple cycles with coarse-grained OpenMP root parallelism.
 *
 * The duplicate-avoidance rule matches the sequential Johnson baseline: each
 * directed cycle is counted only from its smallest compact vertex id. Work is
 * split by root vertex and each worker accumulates into a local histogram
 * before the final merge.
 *
 * @param graph CSR/CSC graph view.
 * @param requested_threads Requested OpenMP thread count.
 * @param max_cycle_length Optional maximum cycle length to count.
 * @return Histogram keyed by cycle length.
 *
 * @throws std::invalid_argument if `requested_threads` is not positive or
 * `max_cycle_length` is less than 2.
 * @throws std::runtime_error if more than one thread is requested in a build
 * without OpenMP support.
 */
[[nodiscard]] CycleHistogram count_simple_cycles_johnson(
    const GraphView& graph,
    int requested_threads,
    std::optional<std::size_t> max_cycle_length = std::nullopt);

}  // namespace cycle_enum::openmp
