#pragma once

#include "cycle_enum/core/graph_view.hpp"
#include "cycle_enum/dynamic/edge_change.hpp"

#include <cstddef>
#include <vector>

/**
 * @file directed_graph.hpp
 * @brief Minimal mutable directed CSR adjacency for incremental updates.
 *
 * The dynamic update works on the directed edge structure only (static simple
 * cycles ignore timestamps). This compact CSR keeps neighbor lists sorted, which
 * makes edge lookup a binary search and keeps traversal deterministic.
 */

namespace cycle_enum::dynamic {

/**
 * @brief Directed graph in CSR form with sorted neighbor lists.
 */
struct DirectedGraph {
  std::size_t vertex_count = 0; ///< Number of vertices.
  std::vector<std::size_t> offsets; ///< CSR offsets, size `vertex_count + 1`.
  std::vector<VertexId> neighbors; ///< Sorted out-neighbors per vertex.
};

/**
 * @brief Build a directed graph from the distinct directed edges of a view.
 */
[[nodiscard]] DirectedGraph build_directed_graph(const GraphView& view);

/**
 * @brief Return a copy of `base` with a batch's deletions removed and
 * insertions added.
 */
[[nodiscard]] DirectedGraph apply_batch(const DirectedGraph& base,
                                        const EdgeBatch& batch);

/**
 * @brief Return true when the directed edge `(source, target)` is present.
 */
[[nodiscard]] bool has_edge(const DirectedGraph& graph,
                            VertexId source,
                            VertexId target);

/**
 * @brief Build a `GraphView` from a directed graph (one dummy timestamp per
 * edge).
 *
 * Used to recompute the post-batch graph with the baseline counter for
 * validation and for recompute-versus-update comparisons. Timestamps are
 * irrelevant for static simple cycles.
 */
[[nodiscard]] GraphView to_graph_view(const DirectedGraph& graph);

}  // namespace cycle_enum::dynamic
