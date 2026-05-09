#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

/**
 * @file graph.hpp
 * @brief Temporal graph parsing and compact host representation.
 */

namespace cycle_enum {

using ExternalVertexId = std::int64_t; ///< Vertex id as it appears in input.
using VertexId = std::uint32_t;        ///< Compact zero-based vertex id.
using Timestamp = std::int64_t;        ///< Edge timestamp type.

/**
 * @brief A grouped directed temporal edge.
 *
 * Multiple input rows with the same source and target are represented as one
 * logical directed edge with a sorted timestamp list. Duplicate timestamps are
 * preserved because they can represent distinct temporal events.
 */
struct TemporalEdge {
  VertexId source = 0; ///< Compact source vertex id.
  VertexId target = 0; ///< Compact target vertex id.
  std::vector<Timestamp> timestamps; ///< Sorted timestamps for this edge.
};

/**
 * @brief Exception thrown when temporal graph parsing fails.
 */
class GraphParseError : public std::runtime_error {
 public:
  /**
   * @brief Construct a parse error with a human-readable message.
   */
  explicit GraphParseError(const std::string& message);
};

/**
 * @brief Compact temporal graph produced by the parser.
 *
 * The graph owns external-to-compact vertex mapping data and grouped temporal
 * edges. CSR and CSC views are built from this representation in a later phase.
 */
class TemporalGraph {
 public:
  /**
   * @brief Construct an empty graph.
   */
  TemporalGraph() = default;

  /**
   * @brief Construct a graph from compact mapping and grouped edges.
   */
  TemporalGraph(std::vector<ExternalVertexId> external_ids,
                std::vector<TemporalEdge> edges);

  /**
   * @brief Return the number of compact vertices.
   */
  [[nodiscard]] std::size_t vertex_count() const noexcept;

  /**
   * @brief Return the number of logical directed edges.
   */
  [[nodiscard]] std::size_t edge_count() const noexcept;

  /**
   * @brief Return the total number of timestamped temporal edge events.
   */
  [[nodiscard]] std::size_t timestamp_count() const noexcept;

  /**
   * @brief Return grouped temporal edges in deterministic source/target order.
   */
  [[nodiscard]] const std::vector<TemporalEdge>& edges() const noexcept;

  /**
   * @brief Return the original external id for a compact vertex.
   *
   * @throws std::out_of_range if the compact id is invalid.
   */
  [[nodiscard]] ExternalVertexId external_id(VertexId vertex) const;

  /**
   * @brief Return the compact id for an external vertex if it exists.
   */
  [[nodiscard]] std::optional<VertexId> compact_id(
      ExternalVertexId external) const noexcept;

  /**
   * @brief Find a grouped edge by compact source and target id.
   */
  [[nodiscard]] const TemporalEdge* find_edge(VertexId source,
                                              VertexId target) const noexcept;

 private:
  std::vector<ExternalVertexId> external_ids_;
  std::vector<TemporalEdge> edges_;
  std::size_t timestamp_count_ = 0;
};

/**
 * @brief Read a directed temporal graph from `source target timestamp` rows.
 *
 * Lines beginning with `#` or `%`, after optional leading whitespace, are
 * ignored. Blank lines are ignored. Self-loops are skipped. Remaining rows must
 * contain exactly three signed integer fields.
 *
 * @throws GraphParseError if the file cannot be opened or a row is malformed.
 */
[[nodiscard]] TemporalGraph read_temporal_graph(
    const std::filesystem::path& path);

}  // namespace cycle_enum
