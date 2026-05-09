#include "cycle_enum/sequential/bruteforce.hpp"

#include <functional>
#include <stdexcept>
#include <vector>

namespace cycle_enum::sequential {

CycleHistogram count_simple_cycles_bruteforce(
    const GraphView& graph,
    const std::optional<std::size_t> max_cycle_length) {
  if (max_cycle_length.has_value() && *max_cycle_length < 2) {
    throw std::invalid_argument("max_cycle_length must be at least 2");
  }

  CycleHistogram histogram;
  std::vector<unsigned char> visited(graph.vertex_count(), 0);
  std::vector<VertexId> path;
  path.reserve(graph.vertex_count());

  for (VertexId root = 0; root < graph.vertex_count(); ++root) {
    path.clear();
    path.push_back(root);
    visited[root] = 1;

    std::function<void(VertexId)> dfs = [&](const VertexId current) {
      const std::size_t begin = graph.outgoing_offsets()[current];
      const std::size_t end = graph.outgoing_offsets()[current + 1];

      for (std::size_t offset = begin; offset < end; ++offset) {
        const VertexId next = graph.outgoing_edges()[offset].vertex;

        if (next == root && path.size() >= 2) {
          histogram.increment(static_cast<CycleHistogram::Length>(path.size()));
          continue;
        }

        if (next <= root || visited[next] != 0) {
          continue;
        }

        if (max_cycle_length.has_value() && path.size() >= *max_cycle_length) {
          continue;
        }

        visited[next] = 1;
        path.push_back(next);
        dfs(next);
        path.pop_back();
        visited[next] = 0;
      }
    };

    dfs(root);
    visited[root] = 0;
  }

  return histogram;
}

}  // namespace cycle_enum::sequential

