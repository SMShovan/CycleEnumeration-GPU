#pragma once

#include "cycle_enum/core/graph_view.hpp"

#include <cstddef>
#include <vector>

/**
 * @file cuda_work_item.hpp
 * @brief Host-side CUDA start-event generation and cycle-union prefiltering.
 */

namespace cycle_enum::cuda {

/**
 * @brief A temporal start event handed to one CUDA thread.
 *
 * The start event is the directed start edge `root -> first_vertex` taken at
 * `start_timestamp`. The device kernel extends it with edges that are strictly
 * later in time and inside the start window.
 */
struct CudaStartEvent {
  VertexId root = 0; ///< Cycle root vertex (the start edge source).
  VertexId first_vertex = 0; ///< Target of the start edge.
  Timestamp start_timestamp = 0; ///< Timestamp of the start edge event.
};

/**
 * @brief Result of building start events, including prefilter accounting.
 */
struct StartEventSet {
  std::vector<CudaStartEvent> events; ///< Surviving start events.
  std::size_t generated = 0; ///< Start events before any prefiltering.
  std::size_t dropped_by_cycle_union = 0; ///< Events removed by the prefilter.
};

/**
 * @brief Build one start event per (directed edge, timestamp) pair.
 *
 * This enumerates every timestamped event as a potential temporal cycle start,
 * matching the work granularity of the naive time-window and temporal kernels.
 */
[[nodiscard]] std::vector<CudaStartEvent> build_start_events(
    const GraphView& graph);

/**
 * @brief Build temporal start events with an optional cycle-union prefilter.
 *
 * When `use_cycle_union` is true, each start event is dropped if its temporal
 * cycle-union set is empty, meaning no vertex can both be reached from the start
 * edge target and return to the root using an edge event after the start
 * timestamp inside the window. The prefilter is conservative: it never removes a
 * start event that can produce a temporal cycle, so device results are
 * unchanged while imbalanced or dead start edges are skipped before launch.
 *
 * @param graph CSR/CSC graph view.
 * @param window_width Inclusive time-window width; must be non-negative.
 * @param use_cycle_union Whether to apply the cycle-union prefilter.
 * @return Surviving start events and prefilter accounting.
 *
 * @throws std::invalid_argument if `window_width` is negative.
 */
[[nodiscard]] StartEventSet build_temporal_start_events(
    const GraphView& graph,
    Timestamp window_width,
    bool use_cycle_union);

}  // namespace cycle_enum::cuda
