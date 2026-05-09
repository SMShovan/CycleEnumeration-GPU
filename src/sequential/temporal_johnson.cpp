#include "cycle_enum/sequential/temporal_johnson.hpp"

#include "cycle_enum/core/timestamp.hpp"

#include <limits>
#include <optional>
#include <stdexcept>
#include <vector>

/**
 * @file temporal_johnson.cpp
 * @brief Naive sequential temporal cycle counter used as the optimization base.
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

void validate_max_cycle_length(
    const std::optional<std::size_t> max_cycle_length) {
  if (max_cycle_length.has_value() && *max_cycle_length < 2) {
    throw std::invalid_argument("max_cycle_length must be at least 2");
  }
}

class TemporalJohnsonSearch {
 public:
  TemporalJohnsonSearch(const GraphView& graph,
                        const Timestamp window_width,
                        std::optional<std::size_t> max_cycle_length)
      : graph_(graph),
        window_width_(window_width),
        max_cycle_length_(max_cycle_length),
        visited_(graph.vertex_count(), 0) {
    validate_max_cycle_length(max_cycle_length_);
    (void)checked_window_end(0, window_width_);
    path_.reserve(graph.vertex_count());
  }

  CycleHistogram run() {
    for (VertexId root = 0; root < graph_.vertex_count(); ++root) {
      root_ = root;
      const std::size_t begin = graph_.outgoing_offsets()[root];
      const std::size_t end = graph_.outgoing_offsets()[root + 1];

      for (std::size_t offset = begin; offset < end; ++offset) {
        const AdjacencyEntry& edge = graph_.outgoing_edges()[offset];
        for (std::size_t timestamp_offset = edge.timestamp_begin;
             timestamp_offset < edge.timestamp_end; ++timestamp_offset) {
          start(edge.vertex, graph_.timestamps()[timestamp_offset]);
        }
      }
    }
    return histogram_;
  }

 private:
  void start(const VertexId first_vertex, const Timestamp start_timestamp) {
    window_end_ = checked_window_end(start_timestamp, window_width_);
    std::fill(visited_.begin(), visited_.end(), 0);
    path_.clear();
    path_.push_back(root_);
    path_.push_back(first_vertex);
    visited_[root_] = 1;
    visited_[first_vertex] = 1;

    dfs(first_vertex, start_timestamp);
  }

  void dfs(const VertexId current, const Timestamp previous_timestamp) {
    const std::size_t begin = graph_.outgoing_offsets()[current];
    const std::size_t end = graph_.outgoing_offsets()[current + 1];

    for (std::size_t offset = begin; offset < end; ++offset) {
      const AdjacencyEntry& edge = graph_.outgoing_edges()[offset];
      const TimestampRange range =
          timestamps_after(graph_.timestamps(), edge.timestamp_begin,
                           edge.timestamp_end, previous_timestamp, window_end_);
      if (range.empty()) {
        continue;
      }

      const VertexId next = edge.vertex;
      if (next == root_ && path_.size() >= 2) {
        histogram_.increment(static_cast<CycleHistogram::Length>(path_.size()),
                             static_cast<CycleHistogram::Count>(range.size()));
        continue;
      }

      if (visited_[next] != 0) {
        continue;
      }

      if (max_cycle_length_.has_value() &&
          path_.size() >= *max_cycle_length_) {
        continue;
      }

      visited_[next] = 1;
      path_.push_back(next);
      for (std::size_t timestamp_offset = range.begin;
           timestamp_offset < range.end; ++timestamp_offset) {
        dfs(next, graph_.timestamps()[timestamp_offset]);
      }
      path_.pop_back();
      visited_[next] = 0;
    }
  }

  const GraphView& graph_;
  Timestamp window_width_;
  std::optional<std::size_t> max_cycle_length_;
  CycleHistogram histogram_;
  std::vector<unsigned char> visited_;
  std::vector<VertexId> path_;
  VertexId root_ = 0;
  Timestamp window_end_ = 0;
};

}  // namespace

CycleHistogram count_temporal_cycles_johnson(
    const GraphView& graph,
    const Timestamp window_width,
    const std::optional<std::size_t> max_cycle_length) {
  return TemporalJohnsonSearch(graph, window_width, max_cycle_length).run();
}

}  // namespace cycle_enum::sequential
