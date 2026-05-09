#include "cycle_enum/cuda/cuda_johnson.hpp"

#include "cycle_enum/cuda/cuda_graph.hpp"
#include "cycle_enum/cuda/cuda_timestamp.hpp"

#include <cuda_runtime_api.h>

#include <limits>
#include <sstream>
#include <stdexcept>
#include <vector>

/**
 * @file cuda_johnson_kernel.cu
 * @brief Naive one-root-per-thread CUDA static cycle counter.
 */

namespace cycle_enum::cuda {
namespace detail {

namespace {

struct DeviceGraphView {
  DeviceOffset vertex_count = 0;
  const DeviceOffset* outgoing_offsets = nullptr;
  const CudaAdjacencyEntry* outgoing_edges = nullptr;
  const Timestamp* timestamps = nullptr;
};

struct CudaStartEvent {
  VertexId root = 0;
  VertexId first_vertex = 0;
  Timestamp start_timestamp = 0;
};

void check_cuda(const cudaError_t error, const char* operation) {
  if (error != cudaSuccess) {
    std::ostringstream message;
    message << operation << " failed: " << cudaGetErrorString(error);
    throw std::runtime_error(message.str());
  }
}

template <typename T>
class DeviceBuffer {
 public:
  DeviceBuffer() = default;

  explicit DeviceBuffer(const std::size_t count) : count_(count) {
    if (count_ > 0) {
      check_cuda(cudaMalloc(reinterpret_cast<void**>(&data_),
                            sizeof(T) * count_),
                 "cudaMalloc");
    }
  }

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  DeviceBuffer(DeviceBuffer&& other) noexcept
      : data_(other.data_), count_(other.count_) {
    other.data_ = nullptr;
    other.count_ = 0;
  }

  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
    if (this != &other) {
      release();
      data_ = other.data_;
      count_ = other.count_;
      other.data_ = nullptr;
      other.count_ = 0;
    }
    return *this;
  }

  ~DeviceBuffer() {
    release();
  }

  [[nodiscard]] T* get() noexcept {
    return data_;
  }

  [[nodiscard]] const T* get() const noexcept {
    return data_;
  }

  void copy_from_host(const std::vector<T>& values, const char* operation) {
    if (values.empty()) {
      return;
    }
    check_cuda(cudaMemcpy(data_, values.data(), sizeof(T) * values.size(),
                          cudaMemcpyHostToDevice),
               operation);
  }

  void copy_to_host(std::vector<T>& values, const char* operation) const {
    if (values.empty()) {
      return;
    }
    check_cuda(cudaMemcpy(values.data(), data_, sizeof(T) * values.size(),
                          cudaMemcpyDeviceToHost),
               operation);
  }

 private:
  void release() noexcept {
    if (data_ != nullptr) {
      (void)cudaFree(data_);
      data_ = nullptr;
      count_ = 0;
    }
  }

  T* data_ = nullptr;
  std::size_t count_ = 0;
};

__device__ bool path_contains(const VertexId* path,
                              const DeviceOffset base,
                              const int depth,
                              const VertexId vertex) {
  for (int index = 0; index < depth; ++index) {
    if (path[base + static_cast<DeviceOffset>(index)] == vertex) {
      return true;
    }
  }
  return false;
}

__global__ void count_roots_kernel(DeviceGraphView graph,
                                   int max_cycle_length,
                                   VertexId* paths,
                                   DeviceOffset* cursors,
                                   unsigned long long* histogram) {
  const DeviceOffset root =
      static_cast<DeviceOffset>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (root >= graph.vertex_count) {
    return;
  }

  const DeviceOffset thread_id = root;
  const DeviceOffset base =
      thread_id * static_cast<DeviceOffset>(max_cycle_length);
  int depth = 1;
  paths[base] = static_cast<VertexId>(root);
  cursors[base] = graph.outgoing_offsets[root];

  while (depth > 0) {
    const VertexId current = paths[base + static_cast<DeviceOffset>(depth - 1)];
    const DeviceOffset end = graph.outgoing_offsets[current + 1];
    DeviceOffset& cursor = cursors[base + static_cast<DeviceOffset>(depth - 1)];

    if (cursor >= end) {
      --depth;
      continue;
    }

    const VertexId next = graph.outgoing_edges[cursor].vertex;
    ++cursor;

    if (next == root && depth >= 2) {
      atomicAdd(&histogram[depth], 1ULL);
      continue;
    }

    if (next <= root || depth >= max_cycle_length ||
        path_contains(paths, base, depth, next)) {
      continue;
    }

    paths[base + static_cast<DeviceOffset>(depth)] = next;
    cursors[base + static_cast<DeviceOffset>(depth)] =
        graph.outgoing_offsets[next];
    ++depth;
  }
}

__global__ void count_time_window_start_events_kernel(
    DeviceGraphView graph,
    const CudaStartEvent* start_events,
    DeviceOffset start_event_count,
    Timestamp window_width,
    int max_cycle_length,
    VertexId* paths,
    DeviceOffset* cursors,
    unsigned long long* histogram) {
  const DeviceOffset event_id =
      static_cast<DeviceOffset>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (event_id >= start_event_count) {
    return;
  }

  const CudaStartEvent event = start_events[event_id];
  const DeviceOffset base =
      event_id * static_cast<DeviceOffset>(max_cycle_length);
  int depth = 2;
  paths[base] = event.root;
  paths[base + 1] = event.first_vertex;
  cursors[base + 1] = graph.outgoing_offsets[event.first_vertex];

  while (depth > 1) {
    const VertexId current = paths[base + static_cast<DeviceOffset>(depth - 1)];
    const DeviceOffset end = graph.outgoing_offsets[current + 1];
    DeviceOffset& cursor = cursors[base + static_cast<DeviceOffset>(depth - 1)];

    if (cursor >= end) {
      --depth;
      continue;
    }

    const CudaAdjacencyEntry edge = graph.outgoing_edges[cursor];
    ++cursor;

    const CudaTimestampStartPolicy policy =
        event.root > current ? CudaTimestampStartPolicy::Inclusive
                             : CudaTimestampStartPolicy::AfterStart;
    if (!has_timestamp_in_window(graph.timestamps, edge.timestamp_begin,
                                 edge.timestamp_end, event.start_timestamp,
                                 window_width, policy)) {
      continue;
    }

    const VertexId next = edge.vertex;
    if (next == event.root && depth >= 2) {
      atomicAdd(&histogram[depth], 1ULL);
      continue;
    }

    if (depth >= max_cycle_length || path_contains(paths, base, depth, next)) {
      continue;
    }

    paths[base + static_cast<DeviceOffset>(depth)] = next;
    cursors[base + static_cast<DeviceOffset>(depth)] =
        graph.outgoing_offsets[next];
    ++depth;
  }
}

std::size_t checked_allocation_count(const DeviceOffset count,
                                     const char* label) {
  if (count > std::numeric_limits<std::size_t>::max()) {
    std::ostringstream message;
    message << label << " exceeds host allocation size";
    throw std::overflow_error(message.str());
  }
  return static_cast<std::size_t>(count);
}

DeviceOffset checked_stack_slots(const DeviceOffset work_count,
                                 const std::size_t max_cycle_length,
                                 const char* label) {
  const DeviceOffset depth = static_cast<DeviceOffset>(max_cycle_length);
  if (depth != 0 && work_count > std::numeric_limits<DeviceOffset>::max() / depth) {
    std::ostringstream message;
    message << label << " stack exceeds CUDA offset range";
    throw std::overflow_error(message.str());
  }
  return work_count * depth;
}

unsigned int checked_grid_blocks(const DeviceOffset work_count,
                                 const unsigned int block_size,
                                 const char* label) {
  const DeviceOffset blocks =
      (work_count + static_cast<DeviceOffset>(block_size) - 1) /
      static_cast<DeviceOffset>(block_size);
  if (blocks > std::numeric_limits<unsigned int>::max()) {
    std::ostringstream message;
    message << label << " exceeds naive CUDA grid capacity";
    throw std::overflow_error(message.str());
  }
  return static_cast<unsigned int>(blocks);
}

std::vector<CudaStartEvent> build_time_window_start_events(
    const CudaGraphData& graph) {
  std::vector<CudaStartEvent> events;
  events.reserve(checked_allocation_count(graph.timestamp_count,
                                          "start event count"));

  for (DeviceOffset root = 0; root < graph.vertex_count; ++root) {
    const DeviceOffset begin = graph.outgoing_offsets[root];
    const DeviceOffset end = graph.outgoing_offsets[root + 1];
    for (DeviceOffset offset = begin; offset < end; ++offset) {
      const CudaAdjacencyEntry& edge = graph.outgoing_edges[offset];
      for (DeviceOffset timestamp_offset = edge.timestamp_begin;
           timestamp_offset < edge.timestamp_end; ++timestamp_offset) {
        events.push_back(CudaStartEvent{
            static_cast<VertexId>(root),
            edge.vertex,
            graph.timestamps[checked_allocation_count(timestamp_offset,
                                                      "timestamp offset")],
        });
      }
    }
  }

  return events;
}

}  // namespace

CycleHistogram count_simple_cycles_johnson_device(
    const CudaGraphData& graph,
    const int device_id,
    const std::size_t max_cycle_length) {
  if (graph.vertex_count == 0 || graph.edge_count == 0) {
    return CycleHistogram{};
  }
  if (max_cycle_length > static_cast<std::size_t>(
                             std::numeric_limits<int>::max())) {
    throw std::overflow_error("max_cycle_length exceeds CUDA kernel limit");
  }

  check_cuda(cudaSetDevice(device_id), "cudaSetDevice");

  DeviceBuffer<DeviceOffset> outgoing_offsets(graph.outgoing_offsets.size());
  DeviceBuffer<CudaAdjacencyEntry> outgoing_edges(graph.outgoing_edges.size());
  DeviceBuffer<unsigned long long> histogram(max_cycle_length + 1);

  const DeviceOffset per_thread_stack =
      checked_stack_slots(graph.vertex_count, max_cycle_length, "path");
  DeviceBuffer<VertexId> paths(checked_allocation_count(per_thread_stack,
                                                        "path stack"));
  DeviceBuffer<DeviceOffset> cursors(checked_allocation_count(per_thread_stack,
                                                              "cursor stack"));

  outgoing_offsets.copy_from_host(graph.outgoing_offsets,
                                  "copy outgoing offsets");
  outgoing_edges.copy_from_host(graph.outgoing_edges, "copy outgoing edges");
  check_cuda(cudaMemset(histogram.get(), 0,
                        sizeof(unsigned long long) * (max_cycle_length + 1)),
             "clear histogram");

  const DeviceGraphView device_graph{
      graph.vertex_count,
      outgoing_offsets.get(),
      outgoing_edges.get(),
      nullptr,
  };

  constexpr unsigned int block_size = 128;
  const unsigned int blocks =
      checked_grid_blocks(graph.vertex_count, block_size, "vertex count");
  count_roots_kernel<<<blocks, block_size>>>(
      device_graph, static_cast<int>(max_cycle_length), paths.get(),
      cursors.get(), histogram.get());
  check_cuda(cudaGetLastError(), "count_roots_kernel launch");
  check_cuda(cudaDeviceSynchronize(), "count_roots_kernel synchronize");

  std::vector<unsigned long long> host_histogram(max_cycle_length + 1, 0);
  histogram.copy_to_host(host_histogram, "copy histogram");

  CycleHistogram result;
  for (std::size_t length = 2; length <= max_cycle_length; ++length) {
    if (host_histogram[length] == 0) {
      continue;
    }
    result.increment(static_cast<CycleHistogram::Length>(length),
                     static_cast<CycleHistogram::Count>(host_histogram[length]));
  }
  return result;
}

CycleHistogram count_time_window_cycles_johnson_device(
    const CudaGraphData& graph,
    const int device_id,
    const Timestamp window_width,
    const std::size_t max_cycle_length) {
  if (graph.vertex_count == 0 || graph.edge_count == 0 ||
      graph.timestamp_count == 0) {
    return CycleHistogram{};
  }
  if (max_cycle_length > static_cast<std::size_t>(
                             std::numeric_limits<int>::max())) {
    throw std::overflow_error("max_cycle_length exceeds CUDA kernel limit");
  }

  const std::vector<CudaStartEvent> start_events =
      build_time_window_start_events(graph);
  if (start_events.empty()) {
    return CycleHistogram{};
  }

  check_cuda(cudaSetDevice(device_id), "cudaSetDevice");

  DeviceBuffer<DeviceOffset> outgoing_offsets(graph.outgoing_offsets.size());
  DeviceBuffer<CudaAdjacencyEntry> outgoing_edges(graph.outgoing_edges.size());
  DeviceBuffer<Timestamp> timestamps(graph.timestamps.size());
  DeviceBuffer<CudaStartEvent> device_start_events(start_events.size());
  DeviceBuffer<unsigned long long> histogram(max_cycle_length + 1);

  const DeviceOffset start_event_count =
      static_cast<DeviceOffset>(start_events.size());
  const DeviceOffset per_thread_stack =
      checked_stack_slots(start_event_count, max_cycle_length, "time-window");
  DeviceBuffer<VertexId> paths(checked_allocation_count(per_thread_stack,
                                                        "path stack"));
  DeviceBuffer<DeviceOffset> cursors(checked_allocation_count(per_thread_stack,
                                                              "cursor stack"));

  outgoing_offsets.copy_from_host(graph.outgoing_offsets,
                                  "copy outgoing offsets");
  outgoing_edges.copy_from_host(graph.outgoing_edges, "copy outgoing edges");
  timestamps.copy_from_host(graph.timestamps, "copy timestamps");
  device_start_events.copy_from_host(start_events, "copy start events");
  check_cuda(cudaMemset(histogram.get(), 0,
                        sizeof(unsigned long long) * (max_cycle_length + 1)),
             "clear histogram");

  const DeviceGraphView device_graph{
      graph.vertex_count,
      outgoing_offsets.get(),
      outgoing_edges.get(),
      timestamps.get(),
  };

  constexpr unsigned int block_size = 128;
  const unsigned int blocks =
      checked_grid_blocks(start_event_count, block_size, "start event count");
  count_time_window_start_events_kernel<<<blocks, block_size>>>(
      device_graph, device_start_events.get(), start_event_count, window_width,
      static_cast<int>(max_cycle_length), paths.get(), cursors.get(),
      histogram.get());
  check_cuda(cudaGetLastError(), "count_time_window_start_events_kernel launch");
  check_cuda(cudaDeviceSynchronize(),
             "count_time_window_start_events_kernel synchronize");

  std::vector<unsigned long long> host_histogram(max_cycle_length + 1, 0);
  histogram.copy_to_host(host_histogram, "copy histogram");

  CycleHistogram result;
  for (std::size_t length = 2; length <= max_cycle_length; ++length) {
    if (host_histogram[length] == 0) {
      continue;
    }
    result.increment(static_cast<CycleHistogram::Length>(length),
                     static_cast<CycleHistogram::Count>(host_histogram[length]));
  }
  return result;
}

}  // namespace detail
}  // namespace cycle_enum::cuda
