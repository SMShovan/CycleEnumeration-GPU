#pragma once

#include "cycle_enum/cuda/cuda_graph.hpp"
#include "cycle_enum/core/graph_view.hpp"

#include <vector>

/**
 * @file cuda_timestamp_groups.hpp
 * @brief Host-side grouping of duplicate edge timestamps for CUDA kernels.
 */

namespace cycle_enum::cuda {

/**
 * @brief Per-edge distinct timestamps with multiplicities.
 *
 * For each outgoing edge slot (indexed exactly like `outgoing_neighbors`), the
 * sorted timestamp list is collapsed into distinct values, each tagged with how
 * many original events shared that value. `edge_group_offsets` is a CSR-style
 * index into `group_values` and `group_counts`, with one extra terminating
 * offset.
 */
struct TimestampGroups {
  std::vector<DeviceOffset> edge_group_offsets; ///< Per-edge group ranges.
  std::vector<Timestamp> group_values; ///< Distinct timestamps, increasing.
  std::vector<DeviceOffset> group_counts; ///< Multiplicity per distinct value.
};

/**
 * @brief Group duplicate timestamps on each outgoing edge.
 *
 * Temporal kernels otherwise branch once per timestamped event, so an edge with
 * many duplicate timestamps spawns many identical device frames. Grouping lets a
 * kernel iterate distinct timestamp values and multiply a cycle's contribution
 * by the group count, which preserves duplicate-event multiplicity while cutting
 * the number of timestamp branches. This is the host half of the hybrid bundling
 * strategy in `cuda_temporal_bundling.md`.
 *
 * @param graph CSR/CSC graph view.
 * @return Grouped timestamps indexed by outgoing edge slot.
 */
[[nodiscard]] TimestampGroups build_outgoing_timestamp_groups(
    const GraphView& graph);

}  // namespace cycle_enum::cuda
