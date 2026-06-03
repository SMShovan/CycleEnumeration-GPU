#include "cycle_enum/dynamic/directed_graph.hpp"
#include "cycle_enum/dynamic/edge_change.hpp"

#include <cuda_runtime_api.h>

#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <vector>

/**
 * @file update_cuda_kernel.cu
 * @brief Device kernel for the single-GPU incremental histogram update.
 *
 * One work item per changed edge enumerates the cycles it owns through an
 * explicit depth-first search that mirrors the host `count_cycles_through_edge`
 * primitive: it walks simple paths from the edge target back to its source,
 * bounded by the maximum cycle length, and refuses to cross a same-phase changed
 * edge with a smaller ownership id. Counts are accumulated per length with
 * atomics. The host runs this once on the initial graph (deletions) and once on
 * the post-batch graph (insertions) and forms the signed delta.
 */

namespace cycle_enum::dynamic {
namespace detail {
namespace {

using DeviceOffset = std::uint64_t;

struct DeviceEdge {
  VertexId source = 0;
  VertexId target = 0;
};

void check_cuda(const cudaError_t error, const char* operation) {
  if (error != cudaSuccess) {
    std::ostringstream message;
    message << operation << " failed: " << cudaGetErrorString(error);
    throw std::runtime_error(message.str());
  }
}

// Binary search the sorted change array for directed edge (source, target).
// Returns the change's ownership id (its index) or -1 when it is not a change.
__device__ int find_change(const DeviceEdge* changes, const int change_count,
                           const VertexId source, const VertexId target) {
  int low = 0;
  int high = change_count - 1;
  while (low <= high) {
    const int mid = (low + high) / 2;
    const VertexId ms = changes[mid].source;
    const VertexId mt = changes[mid].target;
    if (ms < source || (ms == source && mt < target)) {
      low = mid + 1;
    } else if (ms > source || (ms == source && mt > target)) {
      high = mid - 1;
    } else {
      return mid;
    }
  }
  return -1;
}

__global__ void count_owned_cycles_kernel(const DeviceOffset* offsets,
                                          const VertexId* neighbors,
                                          const DeviceEdge* changes,
                                          int change_count,
                                          int max_length,
                                          std::uint32_t vertex_count,
                                          VertexId* path_pool,
                                          DeviceOffset* cursor_pool,
                                          char* visited_pool,
                                          unsigned long long* counts) {
  const int work = blockIdx.x * blockDim.x + threadIdx.x;
  if (work >= change_count) {
    return;
  }

  const VertexId source = changes[work].source;
  const VertexId target = changes[work].target;

  VertexId* path = path_pool + static_cast<std::size_t>(work) * max_length;
  DeviceOffset* cursor =
      cursor_pool + static_cast<std::size_t>(work) * max_length;
  char* visited = visited_pool + static_cast<std::size_t>(work) * vertex_count;

  visited[source] = 1;
  visited[target] = 1;

  int depth = 1;
  path[0] = target;
  cursor[0] = offsets[target];

  while (depth > 0) {
    const VertexId current = path[depth - 1];
    const DeviceOffset end = offsets[current + 1];
    if (cursor[depth - 1] >= end) {
      visited[current] = 0;
      --depth;
      continue;
    }

    const VertexId next = neighbors[cursor[depth - 1]];
    ++cursor[depth - 1];

    const int change_id = find_change(changes, change_count, current, next);
    if (change_id >= 0 && change_id < work) {
      continue; // owned by a smaller-id changed edge
    }

    if (next == source) {
      const int length = depth + 1;
      if (length >= 2 && length <= max_length) {
        atomicAdd(&counts[length], 1ULL);
      }
      continue;
    }

    if (next == target || visited[next] != 0) {
      continue;
    }

    if (depth + 1 < max_length) {
      visited[next] = 1;
      path[depth] = next;
      cursor[depth] = offsets[next];
      ++depth;
    }
  }
}

template <typename T>
T* device_alloc(const std::size_t count) {
  T* pointer = nullptr;
  if (count > 0) {
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&pointer), sizeof(T) * count),
               "cudaMalloc");
  }
  return pointer;
}

}  // namespace

std::vector<unsigned long long> count_owned_cycles_device(
    const DirectedGraph& graph,
    const std::vector<EdgeChange>& phase_changes,
    const std::size_t max_cycle_length,
    const int device_id) {
  std::vector<unsigned long long> counts(max_cycle_length + 1, 0);
  if (phase_changes.empty() || graph.vertex_count == 0) {
    return counts;
  }

  check_cuda(cudaSetDevice(device_id), "cudaSetDevice");

  std::vector<DeviceOffset> offsets(graph.offsets.begin(), graph.offsets.end());
  std::vector<DeviceEdge> changes(phase_changes.size());
  for (std::size_t i = 0; i < phase_changes.size(); ++i) {
    changes[i] = DeviceEdge{phase_changes[i].source, phase_changes[i].target};
  }

  const std::size_t work_count = phase_changes.size();
  const std::size_t stack_slots = work_count * max_cycle_length;
  const std::size_t visited_slots = work_count * graph.vertex_count;

  DeviceOffset* d_offsets = device_alloc<DeviceOffset>(offsets.size());
  VertexId* d_neighbors = device_alloc<VertexId>(graph.neighbors.size());
  DeviceEdge* d_changes = device_alloc<DeviceEdge>(changes.size());
  VertexId* d_path = device_alloc<VertexId>(stack_slots);
  DeviceOffset* d_cursor = device_alloc<DeviceOffset>(stack_slots);
  char* d_visited = device_alloc<char>(visited_slots);
  unsigned long long* d_counts =
      device_alloc<unsigned long long>(counts.size());

  check_cuda(cudaMemcpy(d_offsets, offsets.data(),
                        sizeof(DeviceOffset) * offsets.size(),
                        cudaMemcpyHostToDevice),
             "copy offsets");
  if (!graph.neighbors.empty()) {
    check_cuda(cudaMemcpy(d_neighbors, graph.neighbors.data(),
                          sizeof(VertexId) * graph.neighbors.size(),
                          cudaMemcpyHostToDevice),
               "copy neighbors");
  }
  check_cuda(cudaMemcpy(d_changes, changes.data(),
                        sizeof(DeviceEdge) * changes.size(),
                        cudaMemcpyHostToDevice),
             "copy changes");
  check_cuda(cudaMemset(d_visited, 0, sizeof(char) * visited_slots),
             "clear visited");
  check_cuda(cudaMemset(d_counts, 0, sizeof(unsigned long long) * counts.size()),
             "clear counts");

  constexpr unsigned int block_size = 128;
  const unsigned int blocks =
      static_cast<unsigned int>((work_count + block_size - 1) / block_size);
  count_owned_cycles_kernel<<<blocks, block_size>>>(
      d_offsets, d_neighbors, d_changes, static_cast<int>(work_count),
      static_cast<int>(max_cycle_length),
      static_cast<std::uint32_t>(graph.vertex_count), d_path, d_cursor,
      d_visited, d_counts);
  check_cuda(cudaGetLastError(), "count_owned_cycles_kernel launch");
  check_cuda(cudaDeviceSynchronize(), "count_owned_cycles_kernel synchronize");

  check_cuda(cudaMemcpy(counts.data(), d_counts,
                        sizeof(unsigned long long) * counts.size(),
                        cudaMemcpyDeviceToHost),
             "copy counts");

  (void)cudaFree(d_offsets);
  (void)cudaFree(d_neighbors);
  (void)cudaFree(d_changes);
  (void)cudaFree(d_path);
  (void)cudaFree(d_cursor);
  (void)cudaFree(d_visited);
  (void)cudaFree(d_counts);

  return counts;
}

}  // namespace detail
}  // namespace cycle_enum::dynamic
