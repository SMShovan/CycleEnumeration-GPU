#include "cycle_enum/sequential/johnson.hpp"

#include "cycle_enum/core/timestamp.hpp"

#include <algorithm>
#include <stdexcept>
#include <vector>

namespace cycle_enum::sequential {

namespace {

class JohnsonSearch {
 public:
  JohnsonSearch(const GraphView& graph,
                std::optional<std::size_t> max_cycle_length)
      : graph_(graph),
        max_cycle_length_(max_cycle_length),
        blocked_(graph.vertex_count(), 0),
        blocked_list_(graph.vertex_count()) {
    if (max_cycle_length_.has_value() && *max_cycle_length_ < 2) {
      throw std::invalid_argument("max_cycle_length must be at least 2");
    }
    path_.reserve(graph.vertex_count());
  }

  CycleHistogram run() {
    for (VertexId root = 0; root < graph_.vertex_count(); ++root) {
      root_ = root;
      std::fill(blocked_.begin(), blocked_.end(), 0);
      for (std::vector<VertexId>& list : blocked_list_) {
        list.clear();
      }
      path_.clear();
      circuit(root);
    }
    return histogram_;
  }

 private:
  bool circuit(const VertexId vertex) {
    bool found_cycle = false;
    path_.push_back(vertex);
    blocked_[vertex] = 1;

    const std::size_t begin = graph_.outgoing_offsets()[vertex];
    const std::size_t end = graph_.outgoing_offsets()[vertex + 1];
    for (std::size_t offset = begin; offset < end; ++offset) {
      const VertexId next = graph_.outgoing_edges()[offset].vertex;

      if (next == root_ && path_.size() >= 2) {
        histogram_.increment(static_cast<CycleHistogram::Length>(path_.size()));
        found_cycle = true;
        continue;
      }

      if (next <= root_ || blocked_[next] != 0) {
        continue;
      }

      if (max_cycle_length_.has_value() && path_.size() >= *max_cycle_length_) {
        continue;
      }

      if (circuit(next)) {
        found_cycle = true;
      }
    }

    if (found_cycle) {
      unblock(vertex);
    } else {
      add_to_blocked_lists(vertex);
    }

    path_.pop_back();
    return found_cycle;
  }

  void unblock(const VertexId vertex) {
    blocked_[vertex] = 0;
    std::vector<VertexId>& dependents = blocked_list_[vertex];
    while (!dependents.empty()) {
      const VertexId dependent = dependents.back();
      dependents.pop_back();
      if (blocked_[dependent] != 0) {
        unblock(dependent);
      }
    }
  }

  void add_to_blocked_lists(const VertexId vertex) {
    const std::size_t begin = graph_.outgoing_offsets()[vertex];
    const std::size_t end = graph_.outgoing_offsets()[vertex + 1];
    for (std::size_t offset = begin; offset < end; ++offset) {
      const VertexId next = graph_.outgoing_edges()[offset].vertex;
      if (next <= root_) {
        continue;
      }

      std::vector<VertexId>& dependents = blocked_list_[next];
      if (std::find(dependents.begin(), dependents.end(), vertex) ==
          dependents.end()) {
        dependents.push_back(vertex);
      }
    }
  }

  const GraphView& graph_;
  std::optional<std::size_t> max_cycle_length_;
  CycleHistogram histogram_;
  std::vector<unsigned char> blocked_;
  std::vector<std::vector<VertexId>> blocked_list_;
  std::vector<VertexId> path_;
  VertexId root_ = 0;
};

class JohnsonTimeWindowSearch {
 public:
  JohnsonTimeWindowSearch(const GraphView& graph,
                          Timestamp window_width,
                          std::optional<std::size_t> max_cycle_length)
      : graph_(graph),
        window_width_(window_width),
        max_cycle_length_(max_cycle_length),
        blocked_(graph.vertex_count(), 0),
        blocked_list_(graph.vertex_count()) {
    if (max_cycle_length_.has_value() && *max_cycle_length_ < 2) {
      throw std::invalid_argument("max_cycle_length must be at least 2");
    }
    path_.reserve(graph.vertex_count());
  }

  CycleHistogram run() {
    for (VertexId root = 0; root < graph_.vertex_count(); ++root) {
      root_ = root;
      const std::size_t root_begin = graph_.outgoing_offsets()[root];
      const std::size_t root_end = graph_.outgoing_offsets()[root + 1];

      for (std::size_t start_offset = root_begin; start_offset < root_end;
           ++start_offset) {
        const AdjacencyEntry& start_edge = graph_.outgoing_edges()[start_offset];
        for (std::size_t timestamp_offset = start_edge.timestamp_begin;
             timestamp_offset < start_edge.timestamp_end; ++timestamp_offset) {
          start_timestamp_ = graph_.timestamps()[timestamp_offset];
          reset_state(root);
          circuit(start_edge.vertex);
        }
      }
    }
    return histogram_;
  }

 private:
  void reset_state(const VertexId root) {
    std::fill(blocked_.begin(), blocked_.end(), 0);
    for (std::vector<VertexId>& list : blocked_list_) {
      list.clear();
    }
    path_.clear();
    path_.push_back(root);
  }

  bool circuit(const VertexId vertex) {
    bool found_cycle = false;
    path_.push_back(vertex);
    blocked_[vertex] = 1;

    const std::size_t begin = graph_.outgoing_offsets()[vertex];
    const std::size_t end = graph_.outgoing_offsets()[vertex + 1];
    for (std::size_t offset = begin; offset < end; ++offset) {
      const AdjacencyEntry& edge = graph_.outgoing_edges()[offset];
      const VertexId next = edge.vertex;

      if (!edge_has_timestamp(edge, vertex)) {
        continue;
      }

      if (next == root_ && path_.size() >= 2) {
        histogram_.increment(static_cast<CycleHistogram::Length>(path_.size()));
        found_cycle = true;
        continue;
      }

      if (blocked_[next] != 0) {
        continue;
      }

      if (max_cycle_length_.has_value() && path_.size() >= *max_cycle_length_) {
        continue;
      }

      if (circuit(next)) {
        found_cycle = true;
      }
    }

    if (found_cycle) {
      unblock(vertex);
    } else {
      add_to_blocked_lists(vertex);
    }

    path_.pop_back();
    return found_cycle;
  }

  bool edge_has_timestamp(const AdjacencyEntry& edge,
                          const VertexId current) const {
    const TimestampStartPolicy policy =
        root_ > current ? TimestampStartPolicy::Inclusive
                        : TimestampStartPolicy::AfterStart;
    return has_timestamp_in_window(graph_.timestamps(), edge.timestamp_begin,
                                   edge.timestamp_end, start_timestamp_,
                                   window_width_, policy);
  }

  void unblock(const VertexId vertex) {
    blocked_[vertex] = 0;
    std::vector<VertexId>& dependents = blocked_list_[vertex];
    while (!dependents.empty()) {
      const VertexId dependent = dependents.back();
      dependents.pop_back();
      if (blocked_[dependent] != 0) {
        unblock(dependent);
      }
    }
  }

  void add_to_blocked_lists(const VertexId vertex) {
    const std::size_t begin = graph_.outgoing_offsets()[vertex];
    const std::size_t end = graph_.outgoing_offsets()[vertex + 1];
    for (std::size_t offset = begin; offset < end; ++offset) {
      const AdjacencyEntry& edge = graph_.outgoing_edges()[offset];
      if (edge.vertex == root_ || !edge_has_timestamp(edge, vertex)) {
        continue;
      }

      std::vector<VertexId>& dependents = blocked_list_[edge.vertex];
      if (std::find(dependents.begin(), dependents.end(), vertex) ==
          dependents.end()) {
        dependents.push_back(vertex);
      }
    }
  }

  const GraphView& graph_;
  Timestamp window_width_;
  std::optional<std::size_t> max_cycle_length_;
  CycleHistogram histogram_;
  std::vector<unsigned char> blocked_;
  std::vector<std::vector<VertexId>> blocked_list_;
  std::vector<VertexId> path_;
  VertexId root_ = 0;
  Timestamp start_timestamp_ = 0;
};

}  // namespace

CycleHistogram count_simple_cycles_johnson(
    const GraphView& graph,
    const std::optional<std::size_t> max_cycle_length) {
  return JohnsonSearch(graph, max_cycle_length).run();
}

CycleHistogram count_time_window_cycles_johnson(
    const GraphView& graph,
    const Timestamp window_width,
    const std::optional<std::size_t> max_cycle_length) {
  return JohnsonTimeWindowSearch(graph, window_width, max_cycle_length).run();
}

}  // namespace cycle_enum::sequential
