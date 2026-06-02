#include "cycle_enum/cuda/cuda_work_queue.hpp"

#include "cycle_enum/cuda/cuda_config.hpp"
#include "cycle_enum/cuda/cuda_graph.hpp"

#include <algorithm>
#include <stdexcept>

/**
 * @file cuda_work_queue.cpp
 * @brief Host dispatch and launch planning for the persistent work-queue kernel.
 *
 * The launch planner is host-only arithmetic so it is unit tested on machines
 * without a CUDA device. The device kernel is dispatched only on CUDA builds.
 */

#ifndef CYCLE_ENUM_CUDA_ENABLED
#define CYCLE_ENUM_CUDA_ENABLED 0
#endif

namespace cycle_enum::cuda {

WorkQueueLaunch plan_work_queue_launch(const std::size_t work_item_count,
                                       const unsigned int block_size,
                                       const unsigned int blocks_per_sm,
                                       const unsigned int sm_count) {
  if (block_size == 0 || blocks_per_sm == 0 || sm_count == 0) {
    throw std::invalid_argument(
        "work-queue launch configuration values must be positive");
  }

  if (work_item_count == 0) {
    return WorkQueueLaunch{block_size, 0};
  }

  const std::size_t resident =
      static_cast<std::size_t>(blocks_per_sm) * static_cast<std::size_t>(sm_count);
  const std::size_t grid = std::min(resident, work_item_count);

  return WorkQueueLaunch{block_size, static_cast<unsigned int>(grid)};
}

#if CYCLE_ENUM_CUDA_ENABLED
namespace detail {

[[nodiscard]] CycleHistogram count_simple_cycles_johnson_queue_device(
    const CudaGraphData& graph,
    int device_id,
    std::size_t max_cycle_length);

}  // namespace detail
#endif

CycleHistogram count_simple_cycles_johnson_work_queue(
    const GraphView& graph,
    const int device_id,
    const std::size_t max_cycle_length) {
  if (max_cycle_length < 2) {
    throw std::invalid_argument("max_cycle_length must be at least 2");
  }

#if CYCLE_ENUM_CUDA_ENABLED
  require_device(device_id);
  return detail::count_simple_cycles_johnson_queue_device(
      pack_graph_for_cuda(graph), device_id, max_cycle_length);
#else
  (void)graph;
  (void)device_id;
  throw std::runtime_error("CUDA support is not compiled into this build");
#endif
}

}  // namespace cycle_enum::cuda
