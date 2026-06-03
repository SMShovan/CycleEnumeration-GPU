#include "cycle_enum/dynamic/directed_graph.hpp"

#include "cycle_enum/core/graph.hpp"

#include <algorithm>
#include <vector>

/**
 * @file directed_graph.cpp
 * @brief Minimal mutable directed CSR adjacency for incremental updates.
 */

namespace cycle_enum::dynamic {

namespace {

DirectedGraph build_from_lists(std::vector<std::vector<VertexId>> adjacency) {
  DirectedGraph graph;
  graph.vertex_count = adjacency.size();
  graph.offsets.assign(adjacency.size() + 1, 0);

  for (std::size_t vertex = 0; vertex < adjacency.size(); ++vertex) {
    std::vector<VertexId>& row = adjacency[vertex];
    std::sort(row.begin(), row.end());
    row.erase(std::unique(row.begin(), row.end()), row.end());
    graph.offsets[vertex + 1] = graph.offsets[vertex] + row.size();
  }

  graph.neighbors.reserve(graph.offsets.back());
  for (std::vector<VertexId>& row : adjacency) {
    graph.neighbors.insert(graph.neighbors.end(), row.begin(), row.end());
  }
  return graph;
}

}  // namespace

DirectedGraph build_directed_graph(const GraphView& view) {
  std::vector<std::vector<VertexId>> adjacency(view.vertex_count());
  for (VertexId source = 0; source < view.vertex_count(); ++source) {
    const std::size_t begin = view.outgoing_offsets()[source];
    const std::size_t end = view.outgoing_offsets()[source + 1];
    for (std::size_t offset = begin; offset < end; ++offset) {
      adjacency[source].push_back(view.outgoing_edges()[offset].vertex);
    }
  }
  return build_from_lists(std::move(adjacency));
}

DirectedGraph apply_batch(const DirectedGraph& base, const EdgeBatch& batch) {
  std::vector<std::vector<VertexId>> adjacency(base.vertex_count);

  std::vector<EdgeChange> deletions = batch.deletions;
  std::sort(deletions.begin(), deletions.end());
  const auto is_deleted = [&](const VertexId source, const VertexId target) {
    return std::binary_search(deletions.begin(), deletions.end(),
                              EdgeChange{source, target});
  };

  for (std::size_t source = 0; source < base.vertex_count; ++source) {
    for (std::size_t offset = base.offsets[source];
         offset < base.offsets[source + 1]; ++offset) {
      const VertexId target = base.neighbors[offset];
      if (!is_deleted(static_cast<VertexId>(source), target)) {
        adjacency[source].push_back(target);
      }
    }
  }

  for (const EdgeChange& insertion : batch.insertions) {
    if (insertion.source < base.vertex_count) {
      adjacency[insertion.source].push_back(insertion.target);
    }
  }

  return build_from_lists(std::move(adjacency));
}

bool has_edge(const DirectedGraph& graph, const VertexId source,
              const VertexId target) {
  if (source >= graph.vertex_count) {
    return false;
  }
  const auto begin = graph.neighbors.begin() +
                     static_cast<std::ptrdiff_t>(graph.offsets[source]);
  const auto end = graph.neighbors.begin() +
                   static_cast<std::ptrdiff_t>(graph.offsets[source + 1]);
  return std::binary_search(begin, end, target);
}

GraphView to_graph_view(const DirectedGraph& graph) {
  std::vector<ExternalVertexId> external_ids(graph.vertex_count);
  for (std::size_t vertex = 0; vertex < graph.vertex_count; ++vertex) {
    external_ids[vertex] = static_cast<ExternalVertexId>(vertex);
  }

  std::vector<TemporalEdge> edges;
  edges.reserve(graph.neighbors.size());
  for (std::size_t source = 0; source < graph.vertex_count; ++source) {
    for (std::size_t offset = graph.offsets[source];
         offset < graph.offsets[source + 1]; ++offset) {
      edges.push_back(TemporalEdge{static_cast<VertexId>(source),
                                   graph.neighbors[offset],
                                   std::vector<Timestamp>{0}});
    }
  }

  return build_graph_view(
      TemporalGraph(std::move(external_ids), std::move(edges)));
}

}  // namespace cycle_enum::dynamic
