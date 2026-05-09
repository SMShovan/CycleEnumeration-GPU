#include "cycle_enum/core/graph_view.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace cycle_enum {

namespace {

void ensure_edge_ids_fit(const std::size_t edge_count) {
  if (edge_count > std::numeric_limits<EdgeId>::max()) {
    throw std::overflow_error("graph has more logical edges than EdgeId holds");
  }
}

}  // namespace

GraphView::GraphView(std::size_t vertex_count,
                     std::vector<std::size_t> outgoing_offsets,
                     std::vector<AdjacencyEntry> outgoing_edges,
                     std::vector<std::size_t> incoming_offsets,
                     std::vector<AdjacencyEntry> incoming_edges,
                     std::vector<Timestamp> timestamps)
    : vertex_count_(vertex_count),
      outgoing_offsets_(std::move(outgoing_offsets)),
      outgoing_edges_(std::move(outgoing_edges)),
      incoming_offsets_(std::move(incoming_offsets)),
      incoming_edges_(std::move(incoming_edges)),
      timestamps_(std::move(timestamps)) {}

std::size_t GraphView::vertex_count() const noexcept {
  return vertex_count_;
}

std::size_t GraphView::edge_count() const noexcept {
  return outgoing_edges_.size();
}

std::size_t GraphView::timestamp_count() const noexcept {
  return timestamps_.size();
}

std::size_t GraphView::out_degree(const VertexId vertex) const {
  validate_vertex(vertex);
  return outgoing_offsets_[vertex + 1] - outgoing_offsets_[vertex];
}

std::size_t GraphView::in_degree(const VertexId vertex) const {
  validate_vertex(vertex);
  return incoming_offsets_[vertex + 1] - incoming_offsets_[vertex];
}

const std::vector<std::size_t>& GraphView::outgoing_offsets() const noexcept {
  return outgoing_offsets_;
}

const std::vector<AdjacencyEntry>& GraphView::outgoing_edges() const noexcept {
  return outgoing_edges_;
}

const std::vector<std::size_t>& GraphView::incoming_offsets() const noexcept {
  return incoming_offsets_;
}

const std::vector<AdjacencyEntry>& GraphView::incoming_edges() const noexcept {
  return incoming_edges_;
}

const std::vector<Timestamp>& GraphView::timestamps() const noexcept {
  return timestamps_;
}

void GraphView::validate_vertex(const VertexId vertex) const {
  if (vertex >= vertex_count_) {
    throw std::out_of_range("compact vertex id is out of range");
  }
}

GraphView build_graph_view(const TemporalGraph& graph) {
  ensure_edge_ids_fit(graph.edge_count());

  std::vector<std::vector<AdjacencyEntry>> outgoing_by_vertex(
      graph.vertex_count());
  std::vector<std::vector<AdjacencyEntry>> incoming_by_vertex(
      graph.vertex_count());
  std::vector<Timestamp> timestamps;
  timestamps.reserve(graph.timestamp_count());

  EdgeId edge_id = 0;
  for (const TemporalEdge& edge : graph.edges()) {
    const std::size_t timestamp_begin = timestamps.size();
    timestamps.insert(timestamps.end(), edge.timestamps.begin(),
                      edge.timestamps.end());
    const std::size_t timestamp_end = timestamps.size();

    AdjacencyEntry outgoing_entry{
        edge.target,
        edge_id,
        timestamp_begin,
        timestamp_end,
    };
    AdjacencyEntry incoming_entry{
        edge.source,
        edge_id,
        timestamp_begin,
        timestamp_end,
    };

    outgoing_by_vertex[edge.source].push_back(outgoing_entry);
    incoming_by_vertex[edge.target].push_back(incoming_entry);
    ++edge_id;
  }

  for (std::size_t vertex = 0; vertex < graph.vertex_count(); ++vertex) {
    auto order_entries = [](const AdjacencyEntry& lhs,
                            const AdjacencyEntry& rhs) {
      if (lhs.vertex != rhs.vertex) {
        return lhs.vertex < rhs.vertex;
      }
      return lhs.edge_id < rhs.edge_id;
    };
    std::sort(outgoing_by_vertex[vertex].begin(),
              outgoing_by_vertex[vertex].end(), order_entries);
    std::sort(incoming_by_vertex[vertex].begin(),
              incoming_by_vertex[vertex].end(), order_entries);
  }

  std::vector<std::size_t> outgoing_offsets(graph.vertex_count() + 1, 0);
  std::vector<std::size_t> incoming_offsets(graph.vertex_count() + 1, 0);
  std::vector<AdjacencyEntry> outgoing_edges;
  std::vector<AdjacencyEntry> incoming_edges;
  outgoing_edges.reserve(graph.edge_count());
  incoming_edges.reserve(graph.edge_count());

  for (std::size_t vertex = 0; vertex < graph.vertex_count(); ++vertex) {
    outgoing_offsets[vertex] = outgoing_edges.size();
    outgoing_edges.insert(outgoing_edges.end(), outgoing_by_vertex[vertex].begin(),
                          outgoing_by_vertex[vertex].end());

    incoming_offsets[vertex] = incoming_edges.size();
    incoming_edges.insert(incoming_edges.end(), incoming_by_vertex[vertex].begin(),
                          incoming_by_vertex[vertex].end());
  }

  outgoing_offsets[graph.vertex_count()] = outgoing_edges.size();
  incoming_offsets[graph.vertex_count()] = incoming_edges.size();

  return GraphView(graph.vertex_count(), std::move(outgoing_offsets),
                   std::move(outgoing_edges), std::move(incoming_offsets),
                   std::move(incoming_edges), std::move(timestamps));
}

}  // namespace cycle_enum
