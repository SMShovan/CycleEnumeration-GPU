#include "cycle_enum/sequential/read_tarjan.hpp"

#include <optional>
#include <stdexcept>
#include <vector>

namespace cycle_enum::sequential {

namespace {

class ReadTarjanSearch {
 public:
  ReadTarjanSearch(const GraphView& graph,
                   std::optional<std::size_t> max_cycle_length)
      : graph_(graph), max_cycle_length_(max_cycle_length) {
    if (max_cycle_length_.has_value() && *max_cycle_length_ < 2) {
      throw std::invalid_argument("max_cycle_length must be at least 2");
    }
    current_.reserve(graph.vertex_count());
  }

  CycleHistogram run() {
    for (VertexId root = 0; root < graph_.vertex_count(); ++root) {
      root_ = root;
      current_.clear();
      current_.push_back(root);

      std::vector<unsigned char> blocked(graph_.vertex_count(), 0);
      cycles_read_tarjan(root, blocked, std::nullopt, std::nullopt);
    }
    return histogram_;
  }

 private:
  using Path = std::vector<VertexId>;

  bool find_path_dfs(const VertexId vertex,
                     std::vector<unsigned char>& visited,
                     const std::vector<unsigned char>& blocked,
                     Path& path) const {
    visited[vertex] = 1;

    const std::size_t begin = graph_.outgoing_offsets()[vertex];
    const std::size_t end = graph_.outgoing_offsets()[vertex + 1];
    for (std::size_t offset = begin; offset < end; ++offset) {
      const VertexId next = graph_.outgoing_edges()[offset].vertex;

      if (next == root_) {
        path.clear();
        path.push_back(root_);
        path.push_back(vertex);
        return true;
      }

      if (next <= root_ || visited[next] != 0 || blocked[next] != 0) {
        continue;
      }

      if (find_path_dfs(next, visited, blocked, path)) {
        path.push_back(vertex);
        return true;
      }
    }

    return false;
  }

  bool find_path(const VertexId start,
                 std::vector<unsigned char>& blocked,
                 Path& path) const {
    std::vector<unsigned char> visited(graph_.vertex_count(), 0);
    const bool found = find_path_dfs(start, visited, blocked, path);

    if (!found) {
      for (std::size_t vertex = 0; vertex < visited.size(); ++vertex) {
        if (visited[vertex] != 0) {
          blocked[vertex] = 1;
        }
      }
    }

    return found;
  }

  void cycles_read_tarjan(VertexId edge_vertex,
                          const std::vector<unsigned char>& blocked_state,
                          std::optional<Path> main_path,
                          std::optional<Path> alternate_path) {
    const std::size_t begin = graph_.outgoing_offsets()[edge_vertex];
    const std::size_t end = graph_.outgoing_offsets()[edge_vertex + 1];

    for (std::size_t offset = begin; offset < end; ++offset) {
      const VertexId next = graph_.outgoing_edges()[offset].vertex;
      if (next < root_ || blocked_state[next] != 0) {
        continue;
      }

      Path path;
      if (main_path.has_value() && next == main_path->back()) {
        path = std::move(*main_path);
        main_path.reset();
      } else if (alternate_path.has_value() && next == alternate_path->back()) {
        path = std::move(*alternate_path);
        alternate_path.reset();
      } else {
        std::vector<unsigned char> blocked = blocked_state;
        if (!find_path(next, blocked, path)) {
          continue;
        }
      }

      std::vector<unsigned char> branch_blocked = blocked_state;
      follow_path(edge_vertex, branch_blocked, std::move(path));
    }
  }

  void follow_path(const VertexId edge_vertex,
                   std::vector<unsigned char>& blocked,
                   Path path) {
    const std::size_t restore_size = current_.size();
    std::optional<Path> alternate_path;
    VertexId branch_vertex = edge_vertex;

    while (path.back() != root_ && !alternate_path.has_value()) {
      const VertexId previous = path.back();
      path.pop_back();
      current_.push_back(previous);
      blocked[previous] = 1;
      branch_vertex = previous;

      const std::size_t begin = graph_.outgoing_offsets()[previous];
      const std::size_t end = graph_.outgoing_offsets()[previous + 1];
      for (std::size_t offset = begin; offset < end; ++offset) {
        const VertexId candidate = graph_.outgoing_edges()[offset].vertex;
        if (candidate == path.back() || candidate <= root_ ||
            blocked[candidate] != 0) {
          continue;
        }

        Path found_path;
        if (find_path(candidate, blocked, found_path)) {
          alternate_path = std::move(found_path);
          break;
        }
      }
    }

    if (alternate_path.has_value()) {
      cycles_read_tarjan(branch_vertex, blocked, std::move(path),
                         std::move(alternate_path));
    } else if (!max_cycle_length_.has_value() ||
               current_.size() <= *max_cycle_length_) {
      histogram_.increment(static_cast<CycleHistogram::Length>(current_.size()));
    }

    current_.resize(restore_size);
  }

  const GraphView& graph_;
  std::optional<std::size_t> max_cycle_length_;
  CycleHistogram histogram_;
  std::vector<VertexId> current_;
  VertexId root_ = 0;
};

}  // namespace

CycleHistogram count_simple_cycles_read_tarjan(
    const GraphView& graph,
    const std::optional<std::size_t> max_cycle_length) {
  return ReadTarjanSearch(graph, max_cycle_length).run();
}

}  // namespace cycle_enum::sequential

