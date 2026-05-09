#include "cycle_enum/sequential/bruteforce.hpp"

#include "cycle_enum/core/timestamp.hpp"

#include <functional>
#include <limits>
#include <stdexcept>
#include <vector>

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

void validate_max_cycle_length(
    const std::optional<std::size_t> max_cycle_length) {
  if (max_cycle_length.has_value() && *max_cycle_length < 2) {
    throw std::invalid_argument("max_cycle_length must be at least 2");
  }
}

void validate_window_width(const Timestamp window_width) {
  if (window_width < 0) {
    throw std::invalid_argument("time window width must be non-negative");
  }
}

}  // namespace

CycleHistogram count_simple_cycles_bruteforce(
    const GraphView& graph,
    const std::optional<std::size_t> max_cycle_length) {
  validate_max_cycle_length(max_cycle_length);

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

CycleHistogram count_time_window_cycles_bruteforce(
    const GraphView& graph,
    const Timestamp window_width,
    const std::optional<std::size_t> max_cycle_length) {
  validate_max_cycle_length(max_cycle_length);
  validate_window_width(window_width);

  CycleHistogram histogram;
  std::vector<unsigned char> visited(graph.vertex_count(), 0);
  std::vector<VertexId> path;
  path.reserve(graph.vertex_count());

  auto edge_has_timestamp = [&](const AdjacencyEntry& edge,
                                const VertexId root,
                                const VertexId current,
                                const Timestamp start_timestamp) {
    const TimestampStartPolicy policy =
        root > current ? TimestampStartPolicy::Inclusive
                       : TimestampStartPolicy::AfterStart;
    return has_timestamp_in_window(graph.timestamps(), edge.timestamp_begin,
                                   edge.timestamp_end, start_timestamp,
                                   window_width, policy);
  };

  for (VertexId root = 0; root < graph.vertex_count(); ++root) {
    const std::size_t root_begin = graph.outgoing_offsets()[root];
    const std::size_t root_end = graph.outgoing_offsets()[root + 1];

    for (std::size_t start_offset = root_begin; start_offset < root_end;
         ++start_offset) {
      const AdjacencyEntry& start_edge = graph.outgoing_edges()[start_offset];
      for (std::size_t timestamp_offset = start_edge.timestamp_begin;
           timestamp_offset < start_edge.timestamp_end; ++timestamp_offset) {
        const Timestamp start_timestamp = graph.timestamps()[timestamp_offset];

        path.clear();
        path.push_back(root);
        path.push_back(start_edge.vertex);
        std::fill(visited.begin(), visited.end(), 0);
        visited[root] = 1;
        visited[start_edge.vertex] = 1;

        std::function<void(VertexId)> dfs = [&](const VertexId current) {
          const std::size_t begin = graph.outgoing_offsets()[current];
          const std::size_t end = graph.outgoing_offsets()[current + 1];

          for (std::size_t offset = begin; offset < end; ++offset) {
            const AdjacencyEntry& edge = graph.outgoing_edges()[offset];
            const VertexId next = edge.vertex;

            if (!edge_has_timestamp(edge, root, current, start_timestamp)) {
              continue;
            }

            if (next == root && path.size() >= 2) {
              histogram.increment(
                  static_cast<CycleHistogram::Length>(path.size()));
              continue;
            }

            if (visited[next] != 0) {
              continue;
            }

            if (max_cycle_length.has_value() &&
                path.size() >= *max_cycle_length) {
              continue;
            }

            visited[next] = 1;
            path.push_back(next);
            dfs(next);
            path.pop_back();
            visited[next] = 0;
          }
        };

        dfs(start_edge.vertex);
      }
    }
  }

  return histogram;
}

CycleHistogram count_temporal_cycles_bruteforce(
    const GraphView& graph,
    const Timestamp window_width,
    const std::optional<std::size_t> max_cycle_length) {
  validate_max_cycle_length(max_cycle_length);
  validate_window_width(window_width);

  CycleHistogram histogram;
  std::vector<unsigned char> visited(graph.vertex_count(), 0);
  std::vector<VertexId> path;
  path.reserve(graph.vertex_count());

  for (VertexId root = 0; root < graph.vertex_count(); ++root) {
    const std::size_t root_begin = graph.outgoing_offsets()[root];
    const std::size_t root_end = graph.outgoing_offsets()[root + 1];

    for (std::size_t start_offset = root_begin; start_offset < root_end;
         ++start_offset) {
      const AdjacencyEntry& start_edge = graph.outgoing_edges()[start_offset];

      for (std::size_t timestamp_offset = start_edge.timestamp_begin;
           timestamp_offset < start_edge.timestamp_end; ++timestamp_offset) {
        const Timestamp start_timestamp = graph.timestamps()[timestamp_offset];
        const Timestamp window_end =
            checked_window_end(start_timestamp, window_width);

        path.clear();
        path.push_back(root);
        path.push_back(start_edge.vertex);
        std::fill(visited.begin(), visited.end(), 0);
        visited[root] = 1;
        visited[start_edge.vertex] = 1;

        std::function<void(VertexId, Timestamp)> dfs =
            [&](const VertexId current, const Timestamp previous_timestamp) {
              const std::size_t begin = graph.outgoing_offsets()[current];
              const std::size_t end = graph.outgoing_offsets()[current + 1];

              for (std::size_t offset = begin; offset < end; ++offset) {
                const AdjacencyEntry& edge = graph.outgoing_edges()[offset];
                const VertexId next = edge.vertex;
                const TimestampRange range =
                    timestamps_after(graph.timestamps(), edge.timestamp_begin,
                                     edge.timestamp_end, previous_timestamp,
                                     window_end);

                if (range.empty()) {
                  continue;
                }

                if (next == root && path.size() >= 2) {
                  histogram.increment(
                      static_cast<CycleHistogram::Length>(path.size()),
                      static_cast<CycleHistogram::Count>(range.size()));
                  continue;
                }

                if (visited[next] != 0) {
                  continue;
                }

                if (max_cycle_length.has_value() &&
                    path.size() >= *max_cycle_length) {
                  continue;
                }

                visited[next] = 1;
                path.push_back(next);
                for (std::size_t next_timestamp_offset = range.begin;
                     next_timestamp_offset < range.end;
                     ++next_timestamp_offset) {
                  dfs(next, graph.timestamps()[next_timestamp_offset]);
                }
                path.pop_back();
                visited[next] = 0;
              }
            };

        dfs(start_edge.vertex, start_timestamp);
      }
    }
  }

  return histogram;
}

}  // namespace cycle_enum::sequential
