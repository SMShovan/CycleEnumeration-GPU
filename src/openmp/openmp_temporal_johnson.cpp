#include "cycle_enum/openmp/openmp_temporal_johnson.hpp"

#include "cycle_enum/core/timestamp.hpp"
#include "cycle_enum/openmp/openmp_config.hpp"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <map>
#include <optional>
#include <stdexcept>
#include <vector>

/**
 * @file openmp_temporal_johnson.cpp
 * @brief Root-parallel OpenMP temporal cycle counting baseline.
 */

/**
 * @def CYCLE_ENUM_OPENMP_ENABLED
 * @brief Compile-time flag set by CMake when OpenMP support is linked.
 */
#ifndef CYCLE_ENUM_OPENMP_ENABLED
#define CYCLE_ENUM_OPENMP_ENABLED 0
#endif

#if CYCLE_ENUM_OPENMP_ENABLED
#include <omp.h>
#endif

namespace cycle_enum::openmp {

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

using Count = CycleHistogram::Count;

struct TimestampCount {
  Timestamp timestamp = 0;
  Count count = 0;
};

using TimestampBundle = std::vector<TimestampCount>;

void add_count(Count& target, const Count value) {
  if (value > std::numeric_limits<Count>::max() - target) {
    throw std::overflow_error("temporal path count overflow");
  }
  target += value;
}

Count multiply_count(const Count count, const std::size_t multiplier) {
  if (count == 0 || multiplier == 0) {
    return 0;
  }
  if (multiplier > std::numeric_limits<Count>::max() / count) {
    throw std::overflow_error("temporal path count overflow");
  }
  return count * static_cast<Count>(multiplier);
}

class TemporalJohnsonWorker {
 public:
  TemporalJohnsonWorker(const GraphView& graph,
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

  void count_root(const VertexId root) {
    root_ = root;
    const std::size_t begin = graph_.outgoing_offsets()[root];
    const std::size_t end = graph_.outgoing_offsets()[root + 1];

    for (std::size_t offset = begin; offset < end; ++offset) {
      const AdjacencyEntry& edge = graph_.outgoing_edges()[offset];
      std::map<Timestamp, Count> start_counts;
      for (std::size_t timestamp_offset = edge.timestamp_begin;
           timestamp_offset < edge.timestamp_end; ++timestamp_offset) {
        add_count(start_counts[graph_.timestamps()[timestamp_offset]], 1);
      }

      for (const auto& [timestamp, count] : start_counts) {
        start(edge.vertex, timestamp, count);
      }
    }
  }

  [[nodiscard]] const TemporalJohnsonResult& result() const noexcept {
    return result_;
  }

 private:
  void start(const VertexId first_vertex,
             const Timestamp start_timestamp,
             const Count start_count) {
    window_end_ = checked_window_end(start_timestamp, window_width_);
    closing_after_.assign(graph_.vertex_count(), std::nullopt);
    std::fill(visited_.begin(), visited_.end(), 0);
    path_.clear();
    path_.push_back(root_);
    path_.push_back(first_vertex);
    visited_[root_] = 1;
    visited_[first_vertex] = 1;

    dfs(first_vertex, TimestampBundle{{start_timestamp, start_count}});
  }

  bool can_reach_root(const VertexId current,
                      const Timestamp previous_timestamp) {
    const std::optional<Timestamp> closed_after = closing_after_[current];
    if (closed_after.has_value() && previous_timestamp >= *closed_after) {
      return false;
    }

    const std::size_t begin = graph_.outgoing_offsets()[current];
    const std::size_t end = graph_.outgoing_offsets()[current + 1];

    for (std::size_t offset = begin; offset < end; ++offset) {
      const AdjacencyEntry& edge = graph_.outgoing_edges()[offset];
      const TimestampRange range =
          timestamps_after(graph_.timestamps(), edge.timestamp_begin,
                           edge.timestamp_end, previous_timestamp,
                           window_end_);
      if (range.empty()) {
        continue;
      }

      if (edge.vertex == root_) {
        return true;
      }

      for (std::size_t timestamp_offset = range.begin;
           timestamp_offset < range.end; ++timestamp_offset) {
        if (can_reach_root(edge.vertex, graph_.timestamps()[timestamp_offset])) {
          return true;
        }
      }
    }

    if (!closed_after.has_value() || previous_timestamp < *closed_after) {
      closing_after_[current] = previous_timestamp;
    }
    return false;
  }

  [[nodiscard]] TimestampBundle extend_bundle(
      const TimestampBundle& arrivals,
      const AdjacencyEntry& edge) const {
    std::map<Timestamp, Count> next_counts;
    for (const TimestampCount& arrival : arrivals) {
      const TimestampRange range =
          timestamps_after(graph_.timestamps(), edge.timestamp_begin,
                           edge.timestamp_end, arrival.timestamp, window_end_);
      for (std::size_t timestamp_offset = range.begin;
           timestamp_offset < range.end; ++timestamp_offset) {
        add_count(next_counts[graph_.timestamps()[timestamp_offset]],
                  arrival.count);
      }
    }

    TimestampBundle bundle;
    bundle.reserve(next_counts.size());
    for (const auto& [timestamp, count] : next_counts) {
      bundle.push_back(TimestampCount{timestamp, count});
    }
    return bundle;
  }

  [[nodiscard]] Count closing_count(const TimestampBundle& arrivals,
                                    const AdjacencyEntry& edge) const {
    Count total = 0;
    for (const TimestampCount& arrival : arrivals) {
      const TimestampRange range =
          timestamps_after(graph_.timestamps(), edge.timestamp_begin,
                           edge.timestamp_end, arrival.timestamp, window_end_);
      add_count(total, multiply_count(arrival.count, range.size()));
    }
    return total;
  }

  bool dfs(const VertexId current, const TimestampBundle& arrivals) {
    ++result_.stats.dfs_states;
    result_.stats.bundled_arrival_timestamps += arrivals.size();

    if (arrivals.empty()) {
      return false;
    }

    const Timestamp earliest_arrival = arrivals.front().timestamp;
    if (!can_reach_root(current, earliest_arrival)) {
      ++result_.stats.closing_time_prunes;
      return false;
    }

    bool found_cycle = false;
    const std::size_t begin = graph_.outgoing_offsets()[current];
    const std::size_t end = graph_.outgoing_offsets()[current + 1];

    for (std::size_t offset = begin; offset < end; ++offset) {
      const AdjacencyEntry& edge = graph_.outgoing_edges()[offset];
      const VertexId next = edge.vertex;
      if (next == root_ && path_.size() >= 2) {
        const Count count = closing_count(arrivals, edge);
        if (count == 0) {
          continue;
        }
        result_.histogram.increment(
            static_cast<CycleHistogram::Length>(path_.size()), count);
        found_cycle = true;
        continue;
      }

      if (visited_[next] != 0) {
        continue;
      }

      if (max_cycle_length_.has_value() &&
          path_.size() >= *max_cycle_length_) {
        continue;
      }

      const TimestampBundle next_arrivals = extend_bundle(arrivals, edge);
      if (next_arrivals.empty()) {
        continue;
      }

      visited_[next] = 1;
      path_.push_back(next);
      found_cycle = dfs(next, next_arrivals) || found_cycle;
      path_.pop_back();
      visited_[next] = 0;
    }

    return found_cycle;
  }

  const GraphView& graph_;
  Timestamp window_width_;
  std::optional<std::size_t> max_cycle_length_;
  TemporalJohnsonResult result_;
  std::vector<unsigned char> visited_;
  std::vector<VertexId> path_;
  std::vector<std::optional<Timestamp>> closing_after_;
  VertexId root_ = 0;
  Timestamp window_end_ = 0;
};

TemporalJohnsonResult count_single_thread(
    const GraphView& graph,
    const Timestamp window_width,
    const std::optional<std::size_t> max_cycle_length) {
  TemporalJohnsonWorker worker(graph, window_width, max_cycle_length);
  for (VertexId root = 0; root < graph.vertex_count(); ++root) {
    worker.count_root(root);
  }
  return worker.result();
}

#if CYCLE_ENUM_OPENMP_ENABLED
void merge_stats(TemporalJohnsonStats& target,
                 const TemporalJohnsonStats& source) {
  target.dfs_states += source.dfs_states;
  target.closing_time_prunes += source.closing_time_prunes;
  target.bundled_arrival_timestamps += source.bundled_arrival_timestamps;
}
#endif

}  // namespace

CycleHistogram count_temporal_cycles_johnson(
    const GraphView& graph,
    const Timestamp window_width,
    const int requested_threads,
    const std::optional<std::size_t> max_cycle_length) {
  return count_temporal_cycles_johnson_with_stats(
             graph, window_width, requested_threads, max_cycle_length)
      .histogram;
}

TemporalJohnsonResult count_temporal_cycles_johnson_with_stats(
    const GraphView& graph,
    const Timestamp window_width,
    const int requested_threads,
    const std::optional<std::size_t> max_cycle_length) {
  const int threads = resolve_thread_count(requested_threads);

#if CYCLE_ENUM_OPENMP_ENABLED
  if (threads > 1) {
    std::vector<TemporalJohnsonResult> local_results(
        static_cast<std::size_t>(threads));

#pragma omp parallel num_threads(threads)
    {
      const int thread_id = omp_get_thread_num();
      TemporalJohnsonWorker worker(graph, window_width, max_cycle_length);

#pragma omp for schedule(dynamic)
      for (std::ptrdiff_t root = 0;
           root < static_cast<std::ptrdiff_t>(graph.vertex_count()); ++root) {
        worker.count_root(static_cast<VertexId>(root));
      }

      local_results[static_cast<std::size_t>(thread_id)] = worker.result();
    }

    TemporalJohnsonResult merged;
    for (const TemporalJohnsonResult& local : local_results) {
      merged.histogram.merge(local.histogram);
      merge_stats(merged.stats, local.stats);
    }
    return merged;
  }
#endif

  (void)threads;
  return count_single_thread(graph, window_width, max_cycle_length);
}

}  // namespace cycle_enum::openmp
