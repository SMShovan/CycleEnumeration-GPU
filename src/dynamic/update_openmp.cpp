#include "cycle_enum/dynamic/update_openmp.hpp"

#include "cycle_enum/dynamic/cycles_through_edge.hpp"
#include "cycle_enum/dynamic/update_sequential.hpp"
#include "cycle_enum/openmp/openmp_config.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

/**
 * @file update_openmp.cpp
 * @brief OpenMP-parallel incremental update of a static simple-cycle histogram.
 */

#ifndef CYCLE_ENUM_OPENMP_ENABLED
#define CYCLE_ENUM_OPENMP_ENABLED 0
#endif

#if CYCLE_ENUM_OPENMP_ENABLED
#include <omp.h>
#endif

namespace cycle_enum::dynamic {

namespace {

#if CYCLE_ENUM_OPENMP_ENABLED
// Parallelize a phase over its changed edges into a signed delta.
void accumulate_phase_parallel(const DirectedGraph& graph,
                               const std::vector<EdgeChange>& changes,
                               const std::size_t max_length,
                               const long long sign,
                               const int threads,
                               std::vector<long long>& delta) {
  if (changes.empty()) {
    return;
  }

  const ChangedEdgeIndex index(changes);
  std::vector<std::vector<long long>> local(
      static_cast<std::size_t>(threads),
      std::vector<long long>(max_length + 1, 0));

#pragma omp parallel num_threads(threads)
  {
    std::vector<long long>& mine =
        local[static_cast<std::size_t>(omp_get_thread_num())];
    std::vector<std::uint64_t> counts(max_length + 1, 0);

#pragma omp for schedule(dynamic)
    for (std::ptrdiff_t owner = 0;
         owner < static_cast<std::ptrdiff_t>(changes.size()); ++owner) {
      std::fill(counts.begin(), counts.end(), 0);
      count_cycles_through_edge(graph, changes[owner].source,
                                changes[owner].target,
                                static_cast<std::size_t>(owner), index,
                                max_length, counts);
      for (std::size_t length = 2; length <= max_length; ++length) {
        mine[length] += sign * static_cast<long long>(counts[length]);
      }
    }
  }

  for (const std::vector<long long>& partial : local) {
    for (std::size_t length = 2; length <= max_length; ++length) {
      delta[length] += partial[length];
    }
  }
}
#endif  // CYCLE_ENUM_OPENMP_ENABLED

}  // namespace

CycleHistogram update_static_histogram_openmp(
    const DirectedGraph& initial_graph,
    const CycleHistogram& initial_histogram,
    const EdgeBatch& batch,
    const std::size_t max_cycle_length,
    const int requested_threads) {
  if (max_cycle_length < 2) {
    throw std::invalid_argument("max_cycle_length must be at least 2");
  }
  const int threads = openmp::resolve_thread_count(requested_threads);

#if CYCLE_ENUM_OPENMP_ENABLED
  if (threads > 1) {
    EdgeBatch normalized = batch;
    normalize(normalized);

    std::vector<long long> delta(max_cycle_length + 1, 0);
    accumulate_phase_parallel(initial_graph, normalized.deletions,
                              max_cycle_length, -1, threads, delta);
    const DirectedGraph final_graph = apply_batch(initial_graph, normalized);
    accumulate_phase_parallel(final_graph, normalized.insertions,
                              max_cycle_length, +1, threads, delta);

    return apply_histogram_delta(initial_histogram, delta, max_cycle_length);
  }
#endif

  (void)threads;
  return update_static_histogram(initial_graph, initial_histogram, batch,
                                 max_cycle_length);
}

}  // namespace cycle_enum::dynamic
