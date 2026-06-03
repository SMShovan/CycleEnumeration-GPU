#include "cycle_enum/dynamic/update_sequential.hpp"

#include "cycle_enum/dynamic/cycles_through_edge.hpp"

#include <algorithm>
#include <cstdint>
#include <map>
#include <stdexcept>
#include <vector>

/**
 * @file update_sequential.cpp
 * @brief Sequential incremental update of a static simple-cycle histogram.
 */

namespace cycle_enum::dynamic {

namespace {

// Accumulate the per-length counts of cycles owned by each changed edge into the
// signed delta, with the given sign (-1 for deletions, +1 for insertions).
void accumulate_phase(const DirectedGraph& graph,
                      const std::vector<EdgeChange>& changes,
                      const std::size_t max_length,
                      const long long sign,
                      std::vector<long long>& delta) {
  const ChangedEdgeIndex index(changes);
  std::vector<std::uint64_t> counts(max_length + 1, 0);
  for (std::size_t owner_id = 0; owner_id < changes.size(); ++owner_id) {
    std::fill(counts.begin(), counts.end(), 0);
    count_cycles_through_edge(graph, changes[owner_id].source,
                              changes[owner_id].target, owner_id, index,
                              max_length, counts);
    for (std::size_t length = 2; length <= max_length; ++length) {
      delta[length] += sign * static_cast<long long>(counts[length]);
    }
  }
}

}  // namespace

CycleHistogram update_static_histogram(const DirectedGraph& initial_graph,
                                       const CycleHistogram& initial_histogram,
                                       const EdgeBatch& batch,
                                       const std::size_t max_cycle_length) {
  if (max_cycle_length < 2) {
    throw std::invalid_argument("max_cycle_length must be at least 2");
  }

  EdgeBatch normalized = batch;
  normalize(normalized);

  std::vector<long long> delta(max_cycle_length + 1, 0);

  // Delete phase: enumerate in the initial graph and subtract.
  accumulate_phase(initial_graph, normalized.deletions, max_cycle_length, -1,
                   delta);

  // Insert phase: enumerate in the post-batch graph and add.
  const DirectedGraph final_graph = apply_batch(initial_graph, normalized);
  accumulate_phase(final_graph, normalized.insertions, max_cycle_length, +1,
                   delta);

  // Apply deltas to the prior histogram. Lengths beyond the cap are untouched.
  std::map<CycleHistogram::Length, long long> totals;
  for (const auto& [length, count] : initial_histogram.entries()) {
    totals[length] = static_cast<long long>(count);
  }
  for (std::size_t length = 2; length <= max_cycle_length; ++length) {
    if (delta[length] != 0) {
      totals[static_cast<CycleHistogram::Length>(length)] += delta[length];
    }
  }

  CycleHistogram result;
  for (const auto& [length, count] : totals) {
    if (count < 0) {
      throw std::logic_error(
          "dynamic update produced a negative cycle count");
    }
    if (count > 0) {
      result.increment(length, static_cast<CycleHistogram::Count>(count));
    }
  }
  return result;
}

}  // namespace cycle_enum::dynamic
