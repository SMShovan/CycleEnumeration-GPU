#pragma once

#include "cycle_enum/core/graph_view.hpp"
#include "cycle_enum/core/histogram.hpp"
#include "cycle_enum/dynamic/directed_graph.hpp"
#include "cycle_enum/dynamic/edge_change.hpp"

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

/**
 * @brief Backend selector for the incremental update.
 */
enum class UpdateBackend { Sequential, OpenMP, Cuda };

/**
 * @brief Incrementally update a static simple-cycle histogram on a backend.
 *
 * Dispatches to the sequential, OpenMP, or CUDA update. The prior histogram is
 * the known result of `initial_graph` and is supplied as input (it is the cached
 * prior knowledge a streaming caller already has), so this call measures only
 * the incremental work.
 *
 * @param backend Update backend to use.
 * @param initial_graph Directed graph before the batch.
 * @param prior Prior histogram of `initial_graph`.
 * @param batch Edge changes to apply.
 * @param max_cycle_length Inclusive maximum cycle length; must be at least 2.
 * @param threads OpenMP thread count (ignored by other backends).
 * @param device_id CUDA device ordinal (ignored by other backends).
 * @return Histogram of the post-batch graph.
 */
[[nodiscard]] CycleHistogram update_histogram(
    UpdateBackend backend,
    const dynamic::DirectedGraph& initial_graph,
    const CycleHistogram& prior,
    const dynamic::EdgeBatch& batch,
    std::size_t max_cycle_length,
    int threads,
    int device_id);

}  // namespace cycle_enum::engine
