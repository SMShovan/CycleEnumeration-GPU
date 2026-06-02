#include "cycle_enum/cuda/cuda_johnson.hpp"

#include "cycle_enum/cuda/cuda_config.hpp"
#include "cycle_enum/cuda/cuda_graph.hpp"
#include "cycle_enum/cuda/cuda_work_item.hpp"

#include <stdexcept>
#include <vector>

/**
 * @file cuda_johnson.cpp
 * @brief Host dispatch for the naive CUDA static Johnson counter.
 */

/**
 * @def CYCLE_ENUM_CUDA_ENABLED
 * @brief Compile-time flag set by CMake when CUDA runtime support is linked.
 */
#ifndef CYCLE_ENUM_CUDA_ENABLED
#define CYCLE_ENUM_CUDA_ENABLED 0
#endif

namespace cycle_enum::cuda {

namespace {

void validate_max_cycle_length(const std::size_t max_cycle_length) {
  if (max_cycle_length < 2) {
    throw std::invalid_argument("max_cycle_length must be at least 2");
  }
}

void validate_window_width(const Timestamp window_width) {
  if (window_width < 0) {
    throw std::invalid_argument("window_width must be non-negative");
  }
}

}  // namespace

#if CYCLE_ENUM_CUDA_ENABLED
namespace detail {

[[nodiscard]] CycleHistogram count_simple_cycles_johnson_device(
    const CudaGraphData& graph,
    int device_id,
    std::size_t max_cycle_length);

[[nodiscard]] CycleHistogram count_time_window_cycles_johnson_device(
    const CudaGraphData& graph,
    int device_id,
    Timestamp window_width,
    std::size_t max_cycle_length,
    const std::vector<CudaStartEvent>& start_events);

[[nodiscard]] CycleHistogram count_temporal_cycles_johnson_device(
    const CudaGraphData& graph,
    int device_id,
    Timestamp window_width,
    std::size_t max_cycle_length,
    const std::vector<CudaStartEvent>& start_events);

}  // namespace detail
#endif

CycleHistogram count_simple_cycles_johnson(const GraphView& graph,
                                           const int device_id,
                                           const std::size_t max_cycle_length) {
  validate_max_cycle_length(max_cycle_length);

#if CYCLE_ENUM_CUDA_ENABLED
  require_device(device_id);
  return detail::count_simple_cycles_johnson_device(
      pack_graph_for_cuda(graph), device_id, max_cycle_length);
#else
  (void)graph;
  (void)device_id;
  throw std::runtime_error("CUDA support is not compiled into this build");
#endif
}

CycleHistogram count_time_window_cycles_johnson(
    const GraphView& graph,
    const int device_id,
    const Timestamp window_width,
    const std::size_t max_cycle_length) {
  validate_max_cycle_length(max_cycle_length);
  validate_window_width(window_width);

#if CYCLE_ENUM_CUDA_ENABLED
  require_device(device_id);
  // Time-window mode does not require increasing timestamps, so the temporal
  // cycle-union prefilter does not apply; every start event is kept.
  const std::vector<CudaStartEvent> start_events = build_start_events(graph);
  return detail::count_time_window_cycles_johnson_device(
      pack_graph_for_cuda(graph), device_id, window_width, max_cycle_length,
      start_events);
#else
  (void)graph;
  (void)device_id;
  throw std::runtime_error("CUDA support is not compiled into this build");
#endif
}

CycleHistogram count_temporal_cycles_johnson(
    const GraphView& graph,
    const int device_id,
    const Timestamp window_width,
    const std::size_t max_cycle_length) {
  validate_max_cycle_length(max_cycle_length);
  validate_window_width(window_width);

#if CYCLE_ENUM_CUDA_ENABLED
  require_device(device_id);
  // Drop start edges that cannot close a temporal cycle inside the window
  // before paying for a device launch. The prefilter is correctness preserving.
  const StartEventSet start_events =
      build_temporal_start_events(graph, window_width, /*use_cycle_union=*/true);
  return detail::count_temporal_cycles_johnson_device(
      pack_graph_for_cuda(graph), device_id, window_width, max_cycle_length,
      start_events.events);
#else
  (void)graph;
  (void)device_id;
  throw std::runtime_error("CUDA support is not compiled into this build");
#endif
}

}  // namespace cycle_enum::cuda
