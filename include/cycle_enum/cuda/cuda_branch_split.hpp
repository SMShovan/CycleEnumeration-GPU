#pragma once

#include "cycle_enum/core/graph_view.hpp"
#include "cycle_enum/core/histogram.hpp"

#include <cstddef>
#include <optional>
#include <vector>

/**
 * @file cuda_branch_split.hpp
 * @brief Host-side branch splitting for balanced CUDA cycle work items.
 */

namespace cycle_enum::cuda {

/**
 * @brief One continuation work item: a simple path prefix rooted at its cycle
 * root.
 *
 * `prefix.front()` is the cycle root and `prefix.back()` is the vertex the
 * device continues the depth-first search from. The prefix vertices are the
 * already-visited set for that continuation.
 */
struct SplitWorkItem {
  std::vector<VertexId> prefix;
};

/**
 * @brief Result of decomposing the static search into branch-split work items.
 */
struct BranchSplit {
  std::vector<SplitWorkItem> items; ///< Independent continuation work items.
  CycleHistogram closed; ///< Cycles that close within the prefixes.
};

/**
 * @brief Decompose the static simple-cycle search into branch-split work items.
 *
 * A few high-degree roots can own most of the search tree. Splitting each root's
 * search into many short path prefixes turns one heavy work item into many
 * lighter ones that a persistent work queue can balance across the device.
 *
 * The host walks every simple path prefix up to `cutoff_depth` vertices. Cycles
 * that close at length `<= cutoff_depth` are counted directly into `closed`.
 * Each prefix of length `cutoff_depth` that still has an unexplored
 * continuation is emitted as a work item; the device continues the search from
 * its last vertex and finds the longer cycles. Every cycle is therefore counted
 * exactly once, and the smallest-root duplicate-avoidance rule is preserved, so
 * the union of `closed` and the device continuations equals the undecomposed
 * count.
 *
 * @param graph CSR/CSC graph view.
 * @param cutoff_depth Prefix length at which to stop splitting; must be >= 2.
 * @param max_cycle_length Optional inclusive maximum cycle length.
 * @return The prefix-closed histogram and the continuation work items.
 *
 * @throws std::invalid_argument if `cutoff_depth` is less than 2 or
 * `max_cycle_length` is present and less than 2.
 */
[[nodiscard]] BranchSplit split_static_search(
    const GraphView& graph,
    std::size_t cutoff_depth,
    std::optional<std::size_t> max_cycle_length = std::nullopt);

}  // namespace cycle_enum::cuda
