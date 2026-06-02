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
  CudaGraphData packed{
      checked_offset(graph.vertex_count()),
      checked_offset(graph.edge_count()),
      checked_offset(graph.timestamp_count()),
      pack_offsets(graph.outgoing_offsets()),
      pack_adjacency(graph.outgoing_edges()),
      pack_offsets(graph.incoming_offsets()),
      pack_adjacency(graph.incoming_edges()),
      graph.timestamps(),
      {},
      {},
      {},
  };

  // Split the hot outgoing adjacency into a structure-of-arrays layout for
  // coalesced device traversal. Neighbor ids stay 32-bit; timestamp bounds use
  // the 64-bit device offset type so timestamp-heavy graphs remain representable.
  const std::vector<AdjacencyEntry>& outgoing = graph.outgoing_edges();
  packed.outgoing_neighbors.reserve(outgoing.size());
  packed.outgoing_timestamp_begin.reserve(outgoing.size());
  packed.outgoing_timestamp_end.reserve(outgoing.size());
  for (const AdjacencyEntry& entry : outgoing) {
    packed.outgoing_neighbors.push_back(entry.vertex);
    packed.outgoing_timestamp_begin.push_back(
        checked_offset(entry.timestamp_begin));
    packed.outgoing_timestamp_end.push_back(
        checked_offset(entry.timestamp_end));
  }

  return packed;
}

}  // namespace cycle_enum::cuda
