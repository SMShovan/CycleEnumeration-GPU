#pragma once

#include "cycle_enum/dynamic/directed_graph.hpp"
#include "cycle_enum/dynamic/edge_change.hpp"

#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

/**
 * @file cycles_through_edge.hpp
 * @brief Edge-anchored simple-cycle enumeration with edge-id ownership.
 */

namespace cycle_enum::dynamic {

/**
 * @brief Maps a phase's changed edges to their ownership ids.
 *
 * The id of a changed edge is its index in the normalized (sorted) change list.
 * The index is used to enforce that a cycle is counted only by the smallest-id
 * changed edge it contains.
 */
class ChangedEdgeIndex {
 public:
  /**
   * @brief Build the index from a normalized change list (index = id).
   */
  explicit ChangedEdgeIndex(const std::vector<EdgeChange>& changes);

  /**
   * @brief Return true when `(source, target)` is a changed edge whose id is
   * strictly smaller than `owner_id`.
   *
   * A traversal anchored at the edge with `owner_id` must not cross such an
   * edge, because any cycle using it is owned by the smaller-id edge.
   */
  [[nodiscard]] bool forbidden_before(VertexId source, VertexId target,
                                      std::size_t owner_id) const;

  /**
   * @brief Number of indexed changed edges.
   */
  [[nodiscard]] std::size_t size() const noexcept;

 private:
  std::unordered_map<std::uint64_t, std::size_t> id_of_;
};

/**
 * @brief Count simple cycles through the directed edge `(source -> target)`
 * that are owned by it, bucketed by length.
 *
 * The cycle is `source -> target -> ... -> source`. A cycle is owned by this
 * edge iff it contains no same-phase changed edge with a smaller id, which the
 * search enforces by refusing to cross such edges. Each found cycle of length
 * `len` (2 <= len <= `max_length`) adds one to `counts[len]`.
 *
 * @param graph Directed graph to search (the phase's frozen snapshot).
 * @param source Source of the anchored edge.
 * @param target Target of the anchored edge.
 * @param owner_id Ownership id of the anchored edge in its phase.
 * @param phase_changes Ownership index for this phase.
 * @param max_length Inclusive maximum cycle length; must be at least 2.
 * @param counts Output bucket array of size at least `max_length + 1`.
 */
void count_cycles_through_edge(const DirectedGraph& graph,
                               VertexId source,
                               VertexId target,
                               std::size_t owner_id,
                               const ChangedEdgeIndex& phase_changes,
                               std::size_t max_length,
                               std::vector<std::uint64_t>& counts);

}  // namespace cycle_enum::dynamic
