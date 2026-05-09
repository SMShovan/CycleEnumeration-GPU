#pragma once

#include "cycle_enum/core/graph_view.hpp"

#include <cstddef>
#include <vector>

/**
 * @file cycle_union.hpp
 * @brief Sequential temporal cycle-union preprocessing.
 */

namespace cycle_enum::sequential {

/**
 * @brief Start edge event used to compute a temporal cycle-union set.
 */
struct CycleUnionRequest {
  VertexId root = 0; ///< Cycle root vertex.
  VertexId first_vertex = 0; ///< Target of the start edge.
  Timestamp start_timestamp = 0; ///< Timestamp of the start edge event.
  Timestamp window_width = 0; ///< Non-negative time-window width.
};

/**
 * @brief Vertex set that safely contains candidates for a start event's cycles.
 *
 * The current representation is byte-backed rather than bit-packed so it stays
 * easy to inspect in tests and cheap to copy for small sequential fixtures.
 */
class CycleUnion {
 public:
  /**
   * @brief Construct an empty union for `vertex_count` vertices.
   */
  explicit CycleUnion(std::size_t vertex_count = 0);

  /**
   * @brief Include a compact vertex id.
   */
  void include(VertexId vertex);

  /**
   * @brief Return true when a vertex is included.
   */
  [[nodiscard]] bool contains(VertexId vertex) const;

  /**
   * @brief Return the number of vertices represented by the set.
   */
  [[nodiscard]] std::size_t vertex_count() const noexcept;

  /**
   * @brief Return the number of included vertices.
   */
  [[nodiscard]] std::size_t included_count() const noexcept;

  /**
   * @brief Return included vertices in ascending compact-id order.
   */
  [[nodiscard]] std::vector<VertexId> included_vertices() const;

 private:
  void validate_vertex(VertexId vertex) const;

  std::vector<unsigned char> included_;
  std::size_t included_count_ = 0;
};

/**
 * @brief Compute a conservative temporal cycle-union candidate set.
 *
 * A vertex is included when it is reachable from the start edge target and can
 * reach the root using at least one edge event after the start timestamp and
 * inside the start window. Timestamp order between later edges is ignored, so
 * the result is a safe superset of vertices that may appear in valid temporal
 * cycles for this start event.
 */
[[nodiscard]] CycleUnion compute_temporal_cycle_union(
    const GraphView& graph,
    const CycleUnionRequest& request);

}  // namespace cycle_enum::sequential
