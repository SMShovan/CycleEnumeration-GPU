#include "cycle_enum/openmp/openmp_read_tarjan.hpp"

#include "cycle_enum/openmp/openmp_config.hpp"

#include <cstddef>
#include <stdexcept>
#include <vector>

/**
 * @file openmp_read_tarjan.cpp
 * @brief Coarse-grained OpenMP Read-Tarjan-style static cycle baseline.
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

void validate_max_cycle_length(
    const std::optional<std::size_t> max_cycle_length) {
  if (max_cycle_length.has_value() && *max_cycle_length < 2) {
    throw std::invalid_argument("max_cycle_length must be at least 2");
  }
}

void count_root(const GraphView& graph,
                const VertexId root,
                const std::optional<std::size_t> max_cycle_length,
                CycleHistogram& histogram,
                std::vector<unsigned char>& visited,
                std::vector<VertexId>& path) {
  visited.assign(graph.vertex_count(), 0);
  path.clear();
  path.push_back(root);
  visited[root] = 1;

  auto extend = [&](auto& self, const VertexId current) -> void {
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
      self(self, next);
      path.pop_back();
      visited[next] = 0;
    }
  };

  extend(extend, root);
}

CycleHistogram count_single_thread(
    const GraphView& graph,
    const std::optional<std::size_t> max_cycle_length) {
  CycleHistogram histogram;
  std::vector<unsigned char> visited;
  std::vector<VertexId> path;
  path.reserve(graph.vertex_count());

  for (VertexId root = 0; root < graph.vertex_count(); ++root) {
    count_root(graph, root, max_cycle_length, histogram, visited, path);
  }

  return histogram;
}

}  // namespace

CycleHistogram count_simple_cycles_read_tarjan(
    const GraphView& graph,
    const int requested_threads,
    const std::optional<std::size_t> max_cycle_length) {
  validate_max_cycle_length(max_cycle_length);
  const int threads = resolve_thread_count(requested_threads);

#if CYCLE_ENUM_OPENMP_ENABLED
  if (threads > 1) {
    std::vector<CycleHistogram> local_histograms(static_cast<std::size_t>(threads));

#pragma omp parallel num_threads(threads)
    {
      const int thread_id = omp_get_thread_num();
      CycleHistogram& local =
          local_histograms[static_cast<std::size_t>(thread_id)];
      std::vector<unsigned char> visited;
      std::vector<VertexId> path;
      path.reserve(graph.vertex_count());

#pragma omp for schedule(dynamic)
      for (std::ptrdiff_t root = 0;
           root < static_cast<std::ptrdiff_t>(graph.vertex_count()); ++root) {
        count_root(graph, static_cast<VertexId>(root), max_cycle_length, local,
                   visited, path);
      }
    }

    CycleHistogram merged;
    for (const CycleHistogram& local : local_histograms) {
      merged.merge(local);
    }
    return merged;
  }
#endif

  (void)threads;
  return count_single_thread(graph, max_cycle_length);
}

}  // namespace cycle_enum::openmp
