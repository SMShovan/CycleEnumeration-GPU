#include "cycle_enum/openmp/openmp_read_tarjan_tasks.hpp"

#include "cycle_enum/openmp/openmp_config.hpp"

#include <cstddef>
#include <stdexcept>
#include <vector>

/**
 * @file openmp_read_tarjan_tasks.cpp
 * @brief Fine-grained OpenMP task experiment for Read-Tarjan cycle counting.
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

void validate_arguments(const std::optional<std::size_t> max_cycle_length,
                        const std::size_t task_cutoff_depth) {
  if (max_cycle_length.has_value() && *max_cycle_length < 2) {
    throw std::invalid_argument("max_cycle_length must be at least 2");
  }
  if (task_cutoff_depth == 0) {
    throw std::invalid_argument("task_cutoff_depth must be at least 1");
  }
}

// In-place depth-first subtree search shared by the serial fallback and the
// serial tail of each spawned task. It applies the smallest-root
// duplicate-avoidance rule used by the sequential Read-Tarjan baseline.
void count_subtree_serial(const GraphView& graph,
                          const VertexId root,
                          const std::optional<std::size_t> max_cycle_length,
                          CycleHistogram& histogram,
                          std::vector<unsigned char>& visited,
                          std::vector<VertexId>& path) {
  const VertexId current = path.back();
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
    count_subtree_serial(graph, root, max_cycle_length, histogram, visited, path);
    path.pop_back();
    visited[next] = 0;
  }
}

CycleHistogram count_single_thread(
    const GraphView& graph,
    const std::optional<std::size_t> max_cycle_length) {
  CycleHistogram histogram;
  std::vector<unsigned char> visited;
  std::vector<VertexId> path;
  path.reserve(graph.vertex_count());

  for (VertexId root = 0; root < graph.vertex_count(); ++root) {
    visited.assign(graph.vertex_count(), 0);
    path.clear();
    path.push_back(root);
    visited[root] = 1;
    count_subtree_serial(graph, root, max_cycle_length, histogram, visited, path);
  }

  return histogram;
}

#if CYCLE_ENUM_OPENMP_ENABLED

// Spawn a task per branch while the path is shorter than the cutoff, then hand
// the rest of the subtree to the serial search. Each task receives private
// copies of the path and visited state (copy-on-steal), so concurrent tasks
// never share mutable search state. The graph and the per-thread histogram
// array are passed by pointer so the implicit firstprivate capture copies the
// pointer rather than the pointed-to data, which keeps every task writing into
// the same shared histogram array.
void task_expand(const GraphView* graph,
                 const VertexId root,
                 const std::optional<std::size_t> max_cycle_length,
                 const std::size_t task_cutoff_depth,
                 std::vector<CycleHistogram>* local_histograms,
                 std::vector<VertexId> path,
                 std::vector<unsigned char> visited) {
  if (path.size() >= task_cutoff_depth) {
    CycleHistogram& local =
        (*local_histograms)[static_cast<std::size_t>(omp_get_thread_num())];
    count_subtree_serial(*graph, root, max_cycle_length, local, visited, path);
    return;
  }

  const VertexId current = path.back();
  const std::size_t begin = graph->outgoing_offsets()[current];
  const std::size_t end = graph->outgoing_offsets()[current + 1];

  for (std::size_t offset = begin; offset < end; ++offset) {
    const VertexId next = graph->outgoing_edges()[offset].vertex;

    if (next == root && path.size() >= 2) {
      (*local_histograms)[static_cast<std::size_t>(omp_get_thread_num())]
          .increment(static_cast<CycleHistogram::Length>(path.size()));
      continue;
    }

    if (next <= root || visited[next] != 0) {
      continue;
    }

    if (max_cycle_length.has_value() && path.size() >= *max_cycle_length) {
      continue;
    }

    std::vector<VertexId> child_path = path;
    std::vector<unsigned char> child_visited = visited;
    child_path.push_back(next);
    child_visited[next] = 1;

#pragma omp task firstprivate(child_path, child_visited)
    task_expand(graph, root, max_cycle_length, task_cutoff_depth,
                local_histograms, std::move(child_path),
                std::move(child_visited));
  }
}

#endif  // CYCLE_ENUM_OPENMP_ENABLED

}  // namespace

CycleHistogram count_simple_cycles_read_tarjan_tasks(
    const GraphView& graph,
    const int requested_threads,
    const std::optional<std::size_t> max_cycle_length,
    const std::size_t task_cutoff_depth) {
  validate_arguments(max_cycle_length, task_cutoff_depth);
  const int threads = resolve_thread_count(requested_threads);

#if CYCLE_ENUM_OPENMP_ENABLED
  if (threads > 1) {
    std::vector<CycleHistogram> local_histograms(
        static_cast<std::size_t>(threads));

#pragma omp parallel num_threads(threads)
    {
#pragma omp single nowait
      {
        for (VertexId root = 0; root < graph.vertex_count(); ++root) {
          std::vector<VertexId> path;
          path.reserve(graph.vertex_count());
          path.push_back(root);
          std::vector<unsigned char> visited(graph.vertex_count(), 0);
          visited[root] = 1;

#pragma omp task firstprivate(root, path, visited)
          task_expand(&graph, root, max_cycle_length, task_cutoff_depth,
                      &local_histograms, std::move(path), std::move(visited));
        }
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
  (void)task_cutoff_depth;
  return count_single_thread(graph, max_cycle_length);
}

}  // namespace cycle_enum::openmp
