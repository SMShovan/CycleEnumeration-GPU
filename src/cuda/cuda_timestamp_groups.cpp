#include "cycle_enum/cuda/cuda_timestamp_groups.hpp"

/**
 * @file cuda_timestamp_groups.cpp
 * @brief Host-side grouping of duplicate edge timestamps for CUDA kernels.
 *
 * Host-only so it is unit tested without a CUDA device. The device kernels read
 * the grouped arrays and carry a multiplicity scalar along the search path.
 */

namespace cycle_enum::cuda {

TimestampGroups build_outgoing_timestamp_groups(const GraphView& graph) {
  const std::vector<AdjacencyEntry>& edges = graph.outgoing_edges();
  const std::vector<Timestamp>& timestamps = graph.timestamps();

  TimestampGroups groups;
  groups.edge_group_offsets.reserve(edges.size() + 1);
  groups.edge_group_offsets.push_back(0);

  for (const AdjacencyEntry& edge : edges) {
    std::size_t offset = edge.timestamp_begin;
    while (offset < edge.timestamp_end) {
      const Timestamp value = timestamps[offset];
      DeviceOffset count = 0;
      // The timestamp list is sorted, so equal values are adjacent.
      while (offset < edge.timestamp_end && timestamps[offset] == value) {
        ++count;
        ++offset;
      }
      groups.group_values.push_back(value);
      groups.group_counts.push_back(count);
    }
    groups.edge_group_offsets.push_back(
        static_cast<DeviceOffset>(groups.group_values.size()));
  }

  return groups;
}

}  // namespace cycle_enum::cuda
