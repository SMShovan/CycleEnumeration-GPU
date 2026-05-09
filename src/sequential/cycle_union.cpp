#include "cycle_enum/sequential/cycle_union.hpp"

#include "cycle_enum/core/timestamp.hpp"

#include <limits>
#include <queue>
#include <stdexcept>

/**
 * @file cycle_union.cpp
 * @brief Conservative temporal cycle-union preprocessing implementation.
 */

namespace cycle_enum::sequential {

namespace {

Timestamp checked_window_end(const Timestamp start, const Timestamp width) {
  if (width < 0) {
    throw std::invalid_argument("time window width must be non-negative");
  }
  if (start > std::numeric_limits<Timestamp>::max() - width) {
    throw std::overflow_error("time window end overflows timestamp range");
  }
  return start + width;
}

void validate_request(const GraphView& graph, const CycleUnionRequest& request) {
  (void)checked_window_end(request.start_timestamp, request.window_width);
  if (request.root >= graph.vertex_count()) {
    throw std::out_of_range("cycle-union root vertex is out of range");
  }
  if (request.first_vertex >= graph.vertex_count()) {
    throw std::out_of_range("cycle-union first vertex is out of range");
  }
}

bool edge_has_event_after_start(const GraphView& graph,
                                const AdjacencyEntry& edge,
                                const Timestamp start_timestamp,
                                const Timestamp window_end) {
  return !timestamps_after(graph.timestamps(), edge.timestamp_begin,
                           edge.timestamp_end, start_timestamp, window_end)
              .empty();
}

std::vector<unsigned char> forward_reachable(const GraphView& graph,
                                             const VertexId first_vertex,
                                             const Timestamp start_timestamp,
                                             const Timestamp window_end) {
  std::vector<unsigned char> reachable(graph.vertex_count(), 0);
  std::queue<VertexId> queue;
  reachable[first_vertex] = 1;
  queue.push(first_vertex);

  while (!queue.empty()) {
    const VertexId current = queue.front();
    queue.pop();

    const std::size_t begin = graph.outgoing_offsets()[current];
    const std::size_t end = graph.outgoing_offsets()[current + 1];
    for (std::size_t offset = begin; offset < end; ++offset) {
      const AdjacencyEntry& edge = graph.outgoing_edges()[offset];
      if (!edge_has_event_after_start(graph, edge, start_timestamp,
                                      window_end)) {
        continue;
      }

      const VertexId next = edge.vertex;
      if (reachable[next] == 0) {
        reachable[next] = 1;
        queue.push(next);
      }
    }
  }

  return reachable;
}

std::vector<unsigned char> reverse_reachable_to_root(
    const GraphView& graph,
    const VertexId root,
    const Timestamp start_timestamp,
    const Timestamp window_end) {
  std::vector<unsigned char> reachable(graph.vertex_count(), 0);
  std::queue<VertexId> queue;
  reachable[root] = 1;
  queue.push(root);

  while (!queue.empty()) {
    const VertexId current = queue.front();
    queue.pop();

    const std::size_t begin = graph.incoming_offsets()[current];
    const std::size_t end = graph.incoming_offsets()[current + 1];
    for (std::size_t offset = begin; offset < end; ++offset) {
      const AdjacencyEntry& edge = graph.incoming_edges()[offset];
      if (!edge_has_event_after_start(graph, edge, start_timestamp,
                                      window_end)) {
        continue;
      }

      const VertexId previous = edge.vertex;
      if (reachable[previous] == 0) {
        reachable[previous] = 1;
        queue.push(previous);
      }
    }
  }

  return reachable;
}

}  // namespace

CycleUnion::CycleUnion(const std::size_t vertex_count)
    : included_(vertex_count, 0) {}

void CycleUnion::include(const VertexId vertex) {
  validate_vertex(vertex);
  if (included_[vertex] == 0) {
    included_[vertex] = 1;
    ++included_count_;
  }
}

bool CycleUnion::contains(const VertexId vertex) const {
  validate_vertex(vertex);
  return included_[vertex] != 0;
}

std::size_t CycleUnion::vertex_count() const noexcept {
  return included_.size();
}

std::size_t CycleUnion::included_count() const noexcept {
  return included_count_;
}

std::vector<VertexId> CycleUnion::included_vertices() const {
  std::vector<VertexId> vertices;
  vertices.reserve(included_count_);
  for (std::size_t vertex = 0; vertex < included_.size(); ++vertex) {
    if (included_[vertex] != 0) {
      vertices.push_back(static_cast<VertexId>(vertex));
    }
  }
  return vertices;
}

void CycleUnion::validate_vertex(const VertexId vertex) const {
  if (vertex >= included_.size()) {
    throw std::out_of_range("cycle-union vertex is out of range");
  }
}

CycleUnion compute_temporal_cycle_union(const GraphView& graph,
                                        const CycleUnionRequest& request) {
  validate_request(graph, request);
  const Timestamp window_end =
      checked_window_end(request.start_timestamp, request.window_width);

  const std::vector<unsigned char> forward =
      forward_reachable(graph, request.first_vertex, request.start_timestamp,
                        window_end);
  const std::vector<unsigned char> reverse =
      reverse_reachable_to_root(graph, request.root, request.start_timestamp,
                                window_end);

  CycleUnion cycle_union(graph.vertex_count());
  for (std::size_t vertex = 0; vertex < graph.vertex_count(); ++vertex) {
    if (forward[vertex] != 0 && reverse[vertex] != 0) {
      cycle_union.include(static_cast<VertexId>(vertex));
    }
  }

  return cycle_union;
}

}  // namespace cycle_enum::sequential
