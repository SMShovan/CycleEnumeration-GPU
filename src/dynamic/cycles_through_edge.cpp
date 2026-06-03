#include "cycle_enum/dynamic/cycles_through_edge.hpp"

/**
 * @file cycles_through_edge.cpp
 * @brief Edge-anchored simple-cycle enumeration with edge-id ownership.
 */

namespace cycle_enum::dynamic {

namespace {

std::uint64_t edge_key(const VertexId source, const VertexId target) noexcept {
  return (static_cast<std::uint64_t>(source) << 32) |
         static_cast<std::uint64_t>(target);
}

// Depth-first search for simple paths from `current` back to the anchored
// source `u`. `path_size` is the number of vertices on the path starting at the
// anchored target `v`; closing at `u` yields a cycle of length `path_size + 1`.
void search(const DirectedGraph& graph,
            const VertexId u,
            const VertexId v,
            const std::size_t owner_id,
            const ChangedEdgeIndex& phase_changes,
            const std::size_t max_length,
            const VertexId current,
            const std::size_t path_size,
            std::vector<char>& visited,
            std::vector<std::uint64_t>& counts) {
  const std::size_t begin = graph.offsets[current];
  const std::size_t end = graph.offsets[current + 1];
  for (std::size_t offset = begin; offset < end; ++offset) {
    const VertexId next = graph.neighbors[offset];

    if (phase_changes.forbidden_before(current, next, owner_id)) {
      continue; // owned by a smaller-id changed edge
    }

    if (next == u) {
      const std::size_t length = path_size + 1;
      if (length >= 2 && length <= max_length) {
        counts[length] += 1;
      }
      continue;
    }

    if (next == v || visited[next] != 0) {
      continue;
    }

    if (path_size + 1 < max_length) {
      visited[next] = 1;
      search(graph, u, v, owner_id, phase_changes, max_length, next,
             path_size + 1, visited, counts);
      visited[next] = 0;
    }
  }
}

}  // namespace

ChangedEdgeIndex::ChangedEdgeIndex(const std::vector<EdgeChange>& changes) {
  id_of_.reserve(changes.size());
  for (std::size_t index = 0; index < changes.size(); ++index) {
    id_of_.emplace(edge_key(changes[index].source, changes[index].target),
                   index);
  }
}

bool ChangedEdgeIndex::forbidden_before(const VertexId source,
                                        const VertexId target,
                                        const std::size_t owner_id) const {
  const auto found = id_of_.find(edge_key(source, target));
  return found != id_of_.end() && found->second < owner_id;
}

std::size_t ChangedEdgeIndex::size() const noexcept {
  return id_of_.size();
}

void count_cycles_through_edge(const DirectedGraph& graph,
                               const VertexId source,
                               const VertexId target,
                               const std::size_t owner_id,
                               const ChangedEdgeIndex& phase_changes,
                               const std::size_t max_length,
                               std::vector<std::uint64_t>& counts) {
  if (max_length < 2 || target >= graph.vertex_count ||
      source >= graph.vertex_count) {
    return;
  }

  std::vector<char> visited(graph.vertex_count, 0);
  visited[source] = 1;
  visited[target] = 1;
  search(graph, source, target, owner_id, phase_changes, max_length, target,
         1, visited, counts);
}

}  // namespace cycle_enum::dynamic
