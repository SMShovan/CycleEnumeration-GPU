#pragma once

#include "cycle_enum/core/graph_view.hpp"

#include <cstdint>
#include <vector>

/**
 * @file cuda_graph.hpp
 * @brief Host-side graph packing for CUDA kernels.
 */

namespace cycle_enum::cuda {

/**
 * @brief Offset type used in device-facing CSR and timestamp arrays.
 *
 * Vertex and edge ids stay compact 32-bit values, but offsets use 64 bits so a
 * graph with many timestamped edge events can still be represented safely.
 */
using DeviceOffset = std::uint64_t;

/**
 * @brief Adjacency entry layout copied to CUDA kernels.
 */
struct CudaAdjacencyEntry {
  VertexId vertex = 0; ///< Neighbor vertex id.
  EdgeId edge_id = 0; ///< Logical edge id from the host graph.
  DeviceOffset timestamp_begin = 0; ///< Inclusive timestamp offset.
  DeviceOffset timestamp_end = 0; ///< Exclusive timestamp offset.
};

/**
 * @brief Host-owned packed graph arrays ready for device allocation.
 *
 * The layout mirrors `GraphView` while replacing host `std::size_t` offsets
 * with a fixed-width type. Later CUDA phases can copy these arrays to device
 * memory without depending on platform-specific host pointer sizes.
 */
struct CudaGraphData {
  DeviceOffset vertex_count = 0; ///< Number of compact vertices.
  DeviceOffset edge_count = 0; ///< Number of logical directed edges.
  DeviceOffset timestamp_count = 0; ///< Number of timestamped edge events.
  std::vector<DeviceOffset> outgoing_offsets; ///< CSR outgoing offsets.
  std::vector<CudaAdjacencyEntry> outgoing_edges; ///< CSR adjacency entries.
  std::vector<DeviceOffset> incoming_offsets; ///< CSC incoming offsets.
  std::vector<CudaAdjacencyEntry> incoming_edges; ///< CSC adjacency entries.
  std::vector<Timestamp> timestamps; ///< Flat timestamp storage.

  // Structure-of-arrays view of the hot outgoing adjacency. The traversal
  // kernels read these split arrays so each warp load touches only the field it
  // needs and consecutive lanes read consecutive addresses, which keeps the
  // loads coalesced. They hold the same data as `outgoing_edges`, indexed by the
  // same CSR offset, with neighbor ids kept as compact 32-bit `VertexId`.
  std::vector<VertexId> outgoing_neighbors; ///< SoA target vertex per edge slot.
  std::vector<DeviceOffset> outgoing_timestamp_begin; ///< SoA inclusive offset.
  std::vector<DeviceOffset> outgoing_timestamp_end; ///< SoA exclusive offset.

  // CSC cross-reference used by the blocking backend. `incoming_neighbors[k]` is
  // the source vertex of the k-th CSC in-edge slot, and `incoming_csr_index[k]`
  // is the CSR out-edge slot of that same directed edge, so the kernel can read
  // the B bit (indexed by CSR slot) for each predecessor during unblock.
  std::vector<VertexId> incoming_neighbors; ///< SoA source per CSC in-edge slot.
  std::vector<DeviceOffset> incoming_csr_index; ///< CSR slot per CSC in-edge slot.
};

/**
 * @brief Convert a `GraphView` to CUDA-facing packed arrays.
 */
[[nodiscard]] CudaGraphData pack_graph_for_cuda(const GraphView& graph);

}  // namespace cycle_enum::cuda
