#include "cycle_enum/cuda/cuda_graph.hpp"

#include <limits>
#include <stdexcept>

/**
 * @file cuda_graph.cpp
 * @brief Host-side CSR/CSC packing for CUDA backends.
 */

namespace cycle_enum::cuda {

namespace {

DeviceOffset checked_offset(const std::size_t value) {
  if (value > std::numeric_limits<DeviceOffset>::max()) {
    throw std::overflow_error("graph offset does not fit CUDA offset type");
  }
  return static_cast<DeviceOffset>(value);
}

std::vector<DeviceOffset> pack_offsets(
    const std::vector<std::size_t>& offsets) {
  std::vector<DeviceOffset> packed;
  packed.reserve(offsets.size());
  for (const std::size_t offset : offsets) {
    packed.push_back(checked_offset(offset));
  }
  return packed;
}

std::vector<CudaAdjacencyEntry> pack_adjacency(
    const std::vector<AdjacencyEntry>& entries) {
  std::vector<CudaAdjacencyEntry> packed;
  packed.reserve(entries.size());
  for (const AdjacencyEntry& entry : entries) {
    packed.push_back(CudaAdjacencyEntry{
        entry.vertex,
        entry.edge_id,
        checked_offset(entry.timestamp_begin),
        checked_offset(entry.timestamp_end),
    });
  }
  return packed;
}

}  // namespace

CudaGraphData pack_graph_for_cuda(const GraphView& graph) {
  return CudaGraphData{
      checked_offset(graph.vertex_count()),
      checked_offset(graph.edge_count()),
      checked_offset(graph.timestamp_count()),
      pack_offsets(graph.outgoing_offsets()),
      pack_adjacency(graph.outgoing_edges()),
      pack_offsets(graph.incoming_offsets()),
      pack_adjacency(graph.incoming_edges()),
      graph.timestamps(),
  };
}

}  // namespace cycle_enum::cuda
