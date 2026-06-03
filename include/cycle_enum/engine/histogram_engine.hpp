#pragma once

#include "cycle_enum/core/graph_view.hpp"
#include "cycle_enum/core/histogram.hpp"

#include <cstddef>
#include <optional>

/**
 * @file histogram_engine.hpp
 * @brief Uniform entry points for full recomputation and incremental update.
 *
 * The engine facade gives the CLI and benchmark code one place to obtain a cycle
 * histogram, whether by full recomputation of a graph (the baseline and oracle)
 * or, later, by incrementally updating a known histogram under a batch of edge
 * changes.
 */

namespace cycle_enum::engine {

/**
 * @brief Recompute the static simple-cycle histogram of a graph.
 *
 * This is the baseline path and the oracle the incremental update is checked
 * against. It delegates to the sequential Johnson counter.
 *
 * @param graph CSR/CSC graph view.
 * @param max_cycle_length Optional inclusive maximum cycle length.
 * @return Histogram keyed by cycle length.
 *
 * @throws std::invalid_argument if `max_cycle_length` is present and less than 2.
 */
[[nodiscard]] CycleHistogram count_histogram(
    const GraphView& graph,
    std::optional<std::size_t> max_cycle_length = std::nullopt);

}  // namespace cycle_enum::engine
