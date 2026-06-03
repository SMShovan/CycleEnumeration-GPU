#include "cycle_enum/dynamic/update_cuda.hpp"

#include "cycle_enum/cuda/cuda_config.hpp"
#include "cycle_enum/dynamic/update_sequential.hpp"

#include <cstdint>
#include <stdexcept>
#include <vector>

/**
 * @file update_cuda.cpp
 * @brief Host dispatch for the single-GPU incremental update.
 */

#ifndef CYCLE_ENUM_CUDA_ENABLED
#define CYCLE_ENUM_CUDA_ENABLED 0
#endif

namespace cycle_enum::dynamic {

#if CYCLE_ENUM_CUDA_ENABLED
namespace detail {

// Count, per length, the cycles through each change owned by it, in `graph`.
// `phase_changes` is the normalized (sorted) change list: each entry is a work
// item whose ownership id is its index, and the sorted array is the ownership
// lookup. Returns a vector of size `max_cycle_length + 1`.
[[nodiscard]] std::vector<unsigned long long> count_owned_cycles_device(
    const DirectedGraph& graph,
    const std::vector<EdgeChange>& phase_changes,
    std::size_t max_cycle_length,
    int device_id);

}  // namespace detail
#endif

CycleHistogram update_static_histogram_cuda(
    const DirectedGraph& initial_graph,
    const CycleHistogram& initial_histogram,
    const EdgeBatch& batch,
    const std::size_t max_cycle_length,
    const int device_id) {
  if (max_cycle_length < 2) {
    throw std::invalid_argument("max_cycle_length must be at least 2");
  }

#if CYCLE_ENUM_CUDA_ENABLED
  cuda::require_device(device_id);

  EdgeBatch normalized = batch;
  normalize(normalized);

  const std::vector<unsigned long long> deleted =
      detail::count_owned_cycles_device(initial_graph, normalized.deletions,
                                        max_cycle_length, device_id);

  const DirectedGraph final_graph = apply_batch(initial_graph, normalized);
  const std::vector<unsigned long long> inserted =
      detail::count_owned_cycles_device(final_graph, normalized.insertions,
                                        max_cycle_length, device_id);

  std::vector<long long> delta(max_cycle_length + 1, 0);
  for (std::size_t length = 2; length <= max_cycle_length; ++length) {
    delta[length] = static_cast<long long>(inserted[length]) -
                    static_cast<long long>(deleted[length]);
  }

  return apply_histogram_delta(initial_histogram, delta, max_cycle_length);
#else
  (void)initial_graph;
  (void)initial_histogram;
  (void)batch;
  (void)device_id;
  throw std::runtime_error("CUDA support is not compiled into this build");
#endif
}

}  // namespace cycle_enum::dynamic
