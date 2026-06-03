#pragma once

#include "cycle_enum/core/graph.hpp"

#include <cstddef>
#include <vector>

/**
 * @file edge_change.hpp
 * @brief Edge-change and batch types for incremental histogram updates.
 */

namespace cycle_enum::dynamic {

/**
 * @brief Whether a change inserts or deletes a directed edge.
 */
enum class ChangeKind { Delete, Insert };

/**
 * @brief A directed edge change (static; no timestamp).
 */
struct EdgeChange {
  VertexId source = 0; ///< Compact source vertex id.
  VertexId target = 0; ///< Compact target vertex id.
};

/**
 * @brief Equality by source and target.
 */
inline bool operator==(const EdgeChange& lhs, const EdgeChange& rhs) noexcept {
  return lhs.source == rhs.source && lhs.target == rhs.target;
}

inline bool operator!=(const EdgeChange& lhs, const EdgeChange& rhs) noexcept {
  return !(lhs == rhs);
}

/**
 * @brief Lexicographic order by source then target.
 *
 * After sorting a change list by this order, the index of each entry is its
 * per-phase ownership id: a smaller index means a smaller id.
 */
inline bool operator<(const EdgeChange& lhs, const EdgeChange& rhs) noexcept {
  if (lhs.source != rhs.source) {
    return lhs.source < rhs.source;
  }
  return lhs.target < rhs.target;
}

/**
 * @brief A batch of edge changes split into deletions and insertions.
 *
 * The deletion and insertion lists each carry their own ownership id space,
 * because the delete and insert phases are processed independently.
 */
struct EdgeBatch {
  std::vector<EdgeChange> deletions; ///< Edges removed by the batch.
  std::vector<EdgeChange> insertions; ///< Edges added by the batch.

  /**
   * @brief Total number of edge changes.
   */
  [[nodiscard]] std::size_t total_changes() const noexcept {
    return deletions.size() + insertions.size();
  }
};

/**
 * @brief Sort a change list by `(source, target)` and remove duplicates.
 *
 * The resulting order defines the per-phase ownership ids: entry `i` owns a
 * cycle iff `i` is the smallest index among the change edges the cycle contains.
 */
void sort_and_dedup(std::vector<EdgeChange>& changes);

/**
 * @brief Normalize both lists of a batch with `sort_and_dedup`.
 */
void normalize(EdgeBatch& batch);

}  // namespace cycle_enum::dynamic
