#include "cycle_enum/cuda/cuda_work_item.hpp"

#include "cycle_enum/sequential/cycle_union.hpp"

#include <stdexcept>

/**
 * @file cuda_work_item.cpp
 * @brief Host-side CUDA start-event generation and cycle-union prefiltering.
 *
 * This logic is intentionally host-only so it can be unit tested on machines
 * without a CUDA device. The device kernels consume the resulting start events.
 */

namespace cycle_enum::cuda {

std::vector<CudaStartEvent> build_start_events(const GraphView& graph) {
  std::vector<CudaStartEvent> events;
  events.reserve(graph.timestamp_count());

  for (VertexId root = 0; root < graph.vertex_count(); ++root) {
    const std::size_t begin = graph.outgoing_offsets()[root];
    const std::size_t end = graph.outgoing_offsets()[root + 1];
    for (std::size_t offset = begin; offset < end; ++offset) {
      const AdjacencyEntry& edge = graph.outgoing_edges()[offset];
      for (std::size_t timestamp_offset = edge.timestamp_begin;
           timestamp_offset < edge.timestamp_end; ++timestamp_offset) {
        events.push_back(CudaStartEvent{
            root,
            edge.vertex,
            graph.timestamps()[timestamp_offset],
        });
      }
    }
  }

  return events;
}

StartEventSet build_temporal_start_events(const GraphView& graph,
                                          const Timestamp window_width,
                                          const bool use_cycle_union) {
  if (window_width < 0) {
    throw std::invalid_argument("window_width must be non-negative");
  }

  std::vector<CudaStartEvent> all_events = build_start_events(graph);

  StartEventSet result;
  result.generated = all_events.size();

  if (!use_cycle_union) {
    result.events = std::move(all_events);
    return result;
  }

  result.events.reserve(all_events.size());
  for (const CudaStartEvent& event : all_events) {
    const sequential::CycleUnion candidates =
        sequential::compute_temporal_cycle_union(
            graph,
            sequential::CycleUnionRequest{
                event.root,
                event.first_vertex,
                event.start_timestamp,
                window_width,
            });

    if (candidates.included_count() == 0) {
      ++result.dropped_by_cycle_union;
      continue;
    }

    result.events.push_back(event);
  }

  return result;
}

}  // namespace cycle_enum::cuda
