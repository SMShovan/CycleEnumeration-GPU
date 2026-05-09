#pragma once

#include "cycle_enum/core/graph.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

/**
 * @file graph_view.hpp
 * @brief CSR and CSC views over a compact temporal graph.
 */

namespace cycle_enum {

using EdgeId = std::uint32_t; ///< Compact logical edge id.

/**
 * @brief One adjacency entry in CSR or CSC form.
 *
 * For outgoing adjacency, `vertex` is the target. For incoming adjacency,
 * `vertex` is the source. Timestamp offsets refer to `GraphView::timestamps()`.
 */
struct AdjacencyEntry {
  VertexId vertex = 0;              ///< Neighbor vertex id.
  EdgeId edge_id = 0;               ///< Logical edge id from TemporalGraph.
  std::size_t timestamp_begin = 0;  ///< Inclusive timestamp offset.
  std::size_t timestamp_end = 0;    ///< Exclusive timestamp offset.
};

/**
 * @brief CSR and CSC graph layout with flat timestamp storage.
 *
 * This view is optimized for algorithm traversal. Outgoing offsets and edges
 * form a CSR layout, while incoming offsets and edges form a CSC layout. The
 * same flat timestamp array backs both directions.
 */
class GraphView {
 public:
  /**
   * @brief Construct an empty view.
   */
  GraphView() = default;

  /**
   * @brief Construct a graph view from already-built arrays.
   */
  GraphView(std::size_t vertex_count,
            std::vector<std::size_t> outgoing_offsets,
            std::vector<AdjacencyEntry> outgoing_edges,
            std::vector<std::size_t> incoming_offsets,
            std::vector<AdjacencyEntry> incoming_edges,
            std::vector<Timestamp> timestamps);

  /**
   * @brief Return the number of vertices.
   */
  [[nodiscard]] std::size_t vertex_count() const noexcept;

  /**
   * @brief Return the number of logical directed edges.
   */
  [[nodiscard]] std::size_t edge_count() const noexcept;

  /**
   * @brief Return the number of timestamped events.
   */
  [[nodiscard]] std::size_t timestamp_count() const noexcept;

  /**
   * @brief Return outgoing degree for a compact vertex.
   */
  [[nodiscard]] std::size_t out_degree(VertexId vertex) const;

  /**
   * @brief Return incoming degree for a compact vertex.
   */
  [[nodiscard]] std::size_t in_degree(VertexId vertex) const;

  /**
   * @brief Return CSR outgoing offsets.
   */
  [[nodiscard]] const std::vector<std::size_t>& outgoing_offsets()
      const noexcept;

  /**
   * @brief Return CSR outgoing adjacency entries.
   */
  [[nodiscard]] const std::vector<AdjacencyEntry>& outgoing_edges()
      const noexcept;

  /**
   * @brief Return CSC incoming offsets.
   */
  [[nodiscard]] const std::vector<std::size_t>& incoming_offsets()
      const noexcept;

  /**
   * @brief Return CSC incoming adjacency entries.
   */
  [[nodiscard]] const std::vector<AdjacencyEntry>& incoming_edges()
      const noexcept;

  /**
   * @brief Return flat timestamp storage.
   */
  [[nodiscard]] const std::vector<Timestamp>& timestamps() const noexcept;

 private:
  void validate_vertex(VertexId vertex) const;

  std::size_t vertex_count_ = 0;
  std::vector<std::size_t> outgoing_offsets_;
  std::vector<AdjacencyEntry> outgoing_edges_;
  std::vector<std::size_t> incoming_offsets_;
  std::vector<AdjacencyEntry> incoming_edges_;
  std::vector<Timestamp> timestamps_;
};

/**
 * @brief Build CSR and CSC arrays from a compact temporal graph.
 *
 * @throws std::overflow_error if the logical edge count does not fit EdgeId.
 */
[[nodiscard]] GraphView build_graph_view(const TemporalGraph& graph);

}  // namespace cycle_enum

