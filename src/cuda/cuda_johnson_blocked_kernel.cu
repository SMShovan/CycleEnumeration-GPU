#include "cycle_enum/cuda/cuda_graph.hpp"
#include "cycle_enum/cuda/cuda_profiling.hpp"
#include "cycle_enum/cuda/cuda_work_queue.hpp"

#include <cuda_runtime_api.h>

#include <cstdint>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <vector>

/**
 * @file cuda_johnson_blocked_kernel.cu
 * @brief Persistent work-queue Johnson counter WITH blocked/B pruning and a
 *        per-root uncertainty certificate.
 *
 * One pooled thread claims a root from a global counter and runs an iterative
 * Johnson search with a per-thread `blocked` bitset (n bits) and a per-thread
 * `B` bitset (|E| bits, indexed by CSR out-edge slot). Simplicity is enforced by
 * an explicit on-path check (a scan of the current stack, exactly like the
 * non-blocking kernel); `blocked` is pruning only. Under the length cap the
 * blocked/B bookkeeping is truncated, so a root whose search hits the cap on an
 * otherwise-eligible descent is flagged: its count is then only a lower bound,
 * while every unflagged root is exact. The CPU prototype verified that, with the
 * on-path check, the result never overcounts and is exact on unflagged roots.
 */

namespace cycle_enum::cuda {
namespace detail {
namespace {

using Word = unsigned int;  // per-thread bitset word; bitsets are thread-private.

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
      check_cuda(cudaMalloc(reinterpret_cast<void**>(&data_), sizeof(T) * count_),
                 "cudaMalloc");
    }
  }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  DeviceBuffer(DeviceBuffer&& o) noexcept : data_(o.data_), count_(o.count_) {
    o.data_ = nullptr;
    o.count_ = 0;
  }
  DeviceBuffer& operator=(DeviceBuffer&& o) noexcept {
    if (this != &o) {
      release();
      data_ = o.data_;
      count_ = o.count_;
      o.data_ = nullptr;
      o.count_ = 0;
    }
    return *this;
  }
  ~DeviceBuffer() { release(); }

  [[nodiscard]] T* get() noexcept { return data_; }
  [[nodiscard]] const T* get() const noexcept { return data_; }

  void copy_from_host(const std::vector<T>& values, const char* op) {
    if (values.empty()) return;
    check_cuda(cudaMemcpy(data_, values.data(), sizeof(T) * values.size(),
                          cudaMemcpyHostToDevice),
               op);
  }
  void copy_to_host(std::vector<T>& values, const char* op) const {
    if (values.empty()) return;
    check_cuda(cudaMemcpy(values.data(), data_, sizeof(T) * values.size(),
                          cudaMemcpyDeviceToHost),
               op);
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

// RAII CUDA event timers, mirroring the non-blocking work-queue path so the
// reported kernel/memcpy/total breakdown is comparable. No-ops when disabled.
class ScopedCudaTimers {
 public:
  explicit ScopedCudaTimers(const bool enabled) : enabled_(enabled) {
    if (!enabled_) return;
    create(total_start); create(total_stop);
    create(upload_start); create(upload_stop);
    create(kernel_start); create(kernel_stop);
    create(download_start); create(download_stop);
  }
  ScopedCudaTimers(const ScopedCudaTimers&) = delete;
  ScopedCudaTimers& operator=(const ScopedCudaTimers&) = delete;
  ~ScopedCudaTimers() {
    destroy(total_start); destroy(total_stop);
    destroy(upload_start); destroy(upload_stop);
    destroy(kernel_start); destroy(kernel_stop);
    destroy(download_start); destroy(download_stop);
  }
  void record(cudaEvent_t e) const {
    if (enabled_) check_cuda(cudaEventRecord(e), "cudaEventRecord");
  }
  void fill(CudaTimingsMs& out) const {
    check_cuda(cudaEventSynchronize(total_stop), "cudaEventSynchronize");
    out.kernel_ms = elapsed(kernel_start, kernel_stop);
    out.memcpy_ms = elapsed(upload_start, upload_stop) +
                    elapsed(download_start, download_stop);
    out.total_ms = elapsed(total_start, total_stop);
  }
  cudaEvent_t total_start = nullptr, total_stop = nullptr;
  cudaEvent_t upload_start = nullptr, upload_stop = nullptr;
  cudaEvent_t kernel_start = nullptr, kernel_stop = nullptr;
  cudaEvent_t download_start = nullptr, download_stop = nullptr;

 private:
  static void create(cudaEvent_t& e) { check_cuda(cudaEventCreate(&e), "cudaEventCreate"); }
  static void destroy(cudaEvent_t e) { if (e) (void)cudaEventDestroy(e); }
  static float elapsed(cudaEvent_t a, cudaEvent_t b) {
    float ms = 0.0F;
    check_cuda(cudaEventElapsedTime(&ms, a, b), "cudaEventElapsedTime");
    return ms;
  }
  bool enabled_ = false;
};

// Device view of the graph: CSR out-adjacency plus CSC in-adjacency and the
// CSC->CSR cross map used to read B bits during unblock.
struct BlockedGraphView {
  DeviceOffset vertex_count = 0;
  const DeviceOffset* out_offsets = nullptr;
  const VertexId* out_neighbors = nullptr;
  const DeviceOffset* in_offsets = nullptr;
  const VertexId* in_neighbors = nullptr;
  const DeviceOffset* in_csr_index = nullptr;
};

__device__ inline void set_bit(Word* words, const DeviceOffset i) {
  words[i >> 5] |= (1u << (i & 31u));
}
__device__ inline void clear_bit(Word* words, const DeviceOffset i) {
  words[i >> 5] &= ~(1u << (i & 31u));
}
__device__ inline bool test_bit(const Word* words, const DeviceOffset i) {
  return ((words[i >> 5] >> (i & 31u)) & 1u) != 0u;
}

__device__ inline bool on_path(const VertexId* path, const DeviceOffset base,
                               const int depth, const VertexId vertex) {
  for (int i = 0; i < depth; ++i) {
    if (path[base + static_cast<DeviceOffset>(i)] == vertex) return true;
  }
  return false;
}

__device__ void clear_block_histogram(unsigned long long* bh, const int maxlen) {
  for (int i = threadIdx.x; i <= maxlen; i += blockDim.x) bh[i] = 0ULL;
  __syncthreads();
}
__device__ inline void increment_block_histogram(unsigned long long* bh,
                                                 const int len) {
  atomicAdd(&bh[len], 1ULL);
}
__device__ void flush_block_histogram(unsigned long long* bh,
                                      unsigned long long* global,
                                      const int maxlen) {
  __syncthreads();
  for (int i = threadIdx.x; i <= maxlen; i += blockDim.x) {
    const unsigned long long c = bh[i];
    if (c != 0ULL) atomicAdd(&global[i], c);
  }
}

// Iterative unblock. Removes `start` from `blocked` and cascades to predecessors
// recorded in B (in-edges whose B bit is set). If the work stack would exceed
// its capacity, sets cap_hit and stops cascading: the residual blocking only
// reduces the count (lower bound), and the root is flagged.
__device__ void unblock_vertex(const VertexId start, Word* blocked, Word* bbits,
                               const BlockedGraphView& g, VertexId* stack,
                               const DeviceOffset cap, bool* cap_hit) {
  DeviceOffset top = 0;
  if (cap == 0) { *cap_hit = true; return; }
  stack[top++] = start;
  while (top > 0) {
    const VertexId x = stack[--top];
    if (!test_bit(blocked, static_cast<DeviceOffset>(x))) continue;
    clear_bit(blocked, static_cast<DeviceOffset>(x));
    const DeviceOffset begin = g.in_offsets[x];
    const DeviceOffset end = g.in_offsets[x + 1];
    for (DeviceOffset k = begin; k < end; ++k) {
      const DeviceOffset csr = g.in_csr_index[k];
      if (test_bit(bbits, csr)) {
        clear_bit(bbits, csr);
        const VertexId p = g.in_neighbors[k];
        if (test_bit(blocked, static_cast<DeviceOffset>(p))) {
          if (top < cap) {
            stack[top++] = p;
          } else {
            *cap_hit = true;  // residual block -> undercount, flagged
          }
        }
      }
    }
  }
}

__global__ void count_roots_blocked_queue_kernel(
    BlockedGraphView graph, const int max_cycle_length,
    unsigned long long* work_counter, VertexId* paths, DeviceOffset* cursors,
    unsigned char* founds, Word* blocked_all, const DeviceOffset blocked_words,
    Word* b_all, const DeviceOffset b_words, VertexId* unblock_all,
    const DeviceOffset unblock_cap, unsigned long long* histogram,
    unsigned char* uncertain, unsigned long long* flagged_count) {
  extern __shared__ unsigned long long block_histogram[];
  clear_block_histogram(block_histogram, max_cycle_length);

  const DeviceOffset tid =
      static_cast<DeviceOffset>(blockIdx.x) * blockDim.x + threadIdx.x;
  const DeviceOffset L = static_cast<DeviceOffset>(max_cycle_length);
  const DeviceOffset base = tid * L;
  Word* blocked = blocked_all + tid * blocked_words;
  Word* bbits = b_all + tid * b_words;
  VertexId* ustack = unblock_all + tid * unblock_cap;

  while (true) {
    const DeviceOffset root = atomicAdd(work_counter, 1ULL);
    if (root >= graph.vertex_count) break;

    // Reset per-root state (full clear; a later phase can make this incremental).
    for (DeviceOffset i = 0; i < blocked_words; ++i) blocked[i] = 0u;
    for (DeviceOffset i = 0; i < b_words; ++i) bbits[i] = 0u;
    bool cap_hit = false;

    int depth = 1;
    paths[base] = static_cast<VertexId>(root);
    cursors[base] = graph.out_offsets[root];
    founds[base] = 0;
    set_bit(blocked, root);

    while (depth > 0) {
      const VertexId v = paths[base + static_cast<DeviceOffset>(depth - 1)];
      const DeviceOffset end = graph.out_offsets[v + 1];
      DeviceOffset& cur = cursors[base + static_cast<DeviceOffset>(depth - 1)];

      if (cur >= end) {
        const unsigned char f = founds[base + static_cast<DeviceOffset>(depth - 1)];
        if (depth >= 2) {
          founds[base + static_cast<DeviceOffset>(depth - 2)] |= f;
        }
        if (f != 0) {
          unblock_vertex(v, blocked, bbits, graph, ustack, unblock_cap, &cap_hit);
        } else {
          for (DeviceOffset slot = graph.out_offsets[v]; slot < end; ++slot) {
            set_bit(bbits, slot);  // v joins B[out_neighbor(slot)]
          }
        }
        --depth;
        continue;
      }

      const VertexId w = graph.out_neighbors[cur];
      ++cur;

      if (w == static_cast<VertexId>(root) && depth >= 2) {
        increment_block_histogram(block_histogram, depth);
        founds[base + static_cast<DeviceOffset>(depth - 1)] = 1;
        continue;
      }
      if (w <= static_cast<VertexId>(root)) continue;

      if (static_cast<DeviceOffset>(depth) == L) {
        // Eligible descent suppressed purely by the cap: this root's count may
        // miss a shorter cycle through w, so flag it.
        if (!on_path(paths, base, depth, w) &&
            !test_bit(blocked, static_cast<DeviceOffset>(w))) {
          cap_hit = true;
        }
        continue;
      }
      if (on_path(paths, base, depth, w)) continue;             // simplicity
      if (test_bit(blocked, static_cast<DeviceOffset>(w))) continue;  // pruning

      paths[base + static_cast<DeviceOffset>(depth)] = w;
      cursors[base + static_cast<DeviceOffset>(depth)] = graph.out_offsets[w];
      founds[base + static_cast<DeviceOffset>(depth)] = 0;
      set_bit(blocked, static_cast<DeviceOffset>(w));
      ++depth;
    }

    if (cap_hit) {
      uncertain[root] = 1;
      atomicAdd(flagged_count, 1ULL);
    } else {
      uncertain[root] = 0;
    }
  }

  flush_block_histogram(block_histogram, histogram, max_cycle_length);
}

}  // namespace

CycleHistogram count_simple_cycles_johnson_blocked_queue_device(
    const CudaGraphData& graph, const int device_id,
    const std::size_t max_cycle_length, CudaTimingsMs* const timings,
    std::vector<unsigned char>* const uncertain_out,
    std::uint64_t* const flagged_out) {
  const std::size_t n = static_cast<std::size_t>(graph.vertex_count);
  if (uncertain_out != nullptr) uncertain_out->assign(n, 0);
  if (flagged_out != nullptr) *flagged_out = 0;
  if (graph.vertex_count == 0 || graph.edge_count == 0) return CycleHistogram{};
  if (max_cycle_length >
      static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw std::overflow_error("max_cycle_length exceeds CUDA kernel limit");
  }

  check_cuda(cudaSetDevice(device_id), "cudaSetDevice");
  cudaDeviceProp props{};
  check_cuda(cudaGetDeviceProperties(&props, device_id), "cudaGetDeviceProperties");

  const WorkQueueTuning tuning = work_queue_tuning_from_env();
  WorkQueueLaunch launch = plan_work_queue_launch(
      graph.vertex_count, tuning.block_size, tuning.blocks_per_sm,
      static_cast<unsigned int>(props.multiProcessorCount));
  if (launch.grid_blocks == 0) return CycleHistogram{};

  const DeviceOffset L = static_cast<DeviceOffset>(max_cycle_length);
  const DeviceOffset blocked_words = (graph.vertex_count + 31) / 32;
  const DeviceOffset b_words = (graph.edge_count + 31) / 32;
  // Unblock stack depth is bounded by the number of blocked vertices (<= n).
  // P1 uses n so small graphs are exact; a later phase caps this for memory.
  const DeviceOffset unblock_cap = graph.vertex_count;

  // Clamp the resident grid so the per-thread buffers fit in available memory.
  const std::size_t per_thread =
      static_cast<std::size_t>(L) * sizeof(VertexId) +
      static_cast<std::size_t>(L) * sizeof(DeviceOffset) +
      static_cast<std::size_t>(L) * sizeof(unsigned char) +
      static_cast<std::size_t>(blocked_words) * sizeof(Word) +
      static_cast<std::size_t>(b_words) * sizeof(Word) +
      static_cast<std::size_t>(unblock_cap) * sizeof(VertexId);
  std::size_t free_bytes = 0, total_bytes = 0;
  check_cuda(cudaMemGetInfo(&free_bytes, &total_bytes), "cudaMemGetInfo");
  const std::size_t graph_bytes =
      (graph.outgoing_offsets.size() + graph.incoming_offsets.size()) * sizeof(DeviceOffset) +
      (graph.outgoing_neighbors.size() + graph.incoming_neighbors.size()) * sizeof(VertexId) +
      graph.incoming_csr_index.size() * sizeof(DeviceOffset) +
      (max_cycle_length + 1) * sizeof(unsigned long long) +
      n * sizeof(unsigned char);
  const std::size_t budget =
      free_bytes > graph_bytes ? static_cast<std::size_t>((free_bytes - graph_bytes) * 0.8) : 0;
  const DeviceOffset planned_threads =
      static_cast<DeviceOffset>(launch.grid_blocks) * launch.block_size;
  if (per_thread > 0) {
    const DeviceOffset affordable_threads =
        static_cast<DeviceOffset>(budget / per_thread);
    if (affordable_threads < planned_threads) {
      const DeviceOffset blocks = affordable_threads / launch.block_size;
      if (blocks == 0) {
        throw std::runtime_error(
            "insufficient GPU memory for the blocking backend; reduce "
            "--max-cycle-length or use --blocking no");
      }
      launch.grid_blocks = static_cast<unsigned int>(blocks);
    }
  }
  const DeviceOffset total_threads =
      static_cast<DeviceOffset>(launch.grid_blocks) * launch.block_size;

  ScopedCudaTimers timers(timings != nullptr);
  timers.record(timers.total_start);
  timers.record(timers.upload_start);
  DeviceBuffer<DeviceOffset> d_out_off(graph.outgoing_offsets.size());
  DeviceBuffer<VertexId> d_out_nbr(graph.outgoing_neighbors.size());
  DeviceBuffer<DeviceOffset> d_in_off(graph.incoming_offsets.size());
  DeviceBuffer<VertexId> d_in_nbr(graph.incoming_neighbors.size());
  DeviceBuffer<DeviceOffset> d_in_csr(graph.incoming_csr_index.size());
  d_out_off.copy_from_host(graph.outgoing_offsets, "copy out offsets");
  d_out_nbr.copy_from_host(graph.outgoing_neighbors, "copy out neighbors");
  d_in_off.copy_from_host(graph.incoming_offsets, "copy in offsets");
  d_in_nbr.copy_from_host(graph.incoming_neighbors, "copy in neighbors");
  d_in_csr.copy_from_host(graph.incoming_csr_index, "copy in csr index");
  timers.record(timers.upload_stop);

  DeviceBuffer<VertexId> d_paths(static_cast<std::size_t>(total_threads * L));
  DeviceBuffer<DeviceOffset> d_cursors(static_cast<std::size_t>(total_threads * L));
  DeviceBuffer<unsigned char> d_founds(static_cast<std::size_t>(total_threads * L));
  DeviceBuffer<Word> d_blocked(static_cast<std::size_t>(total_threads * blocked_words));
  DeviceBuffer<Word> d_b(static_cast<std::size_t>(total_threads * b_words));
  DeviceBuffer<VertexId> d_unblock(static_cast<std::size_t>(total_threads * unblock_cap));
  DeviceBuffer<unsigned long long> d_hist(max_cycle_length + 1);
  DeviceBuffer<unsigned long long> d_work(1);
  DeviceBuffer<unsigned char> d_uncertain(n);
  DeviceBuffer<unsigned long long> d_flagged(1);

  check_cuda(cudaMemset(d_hist.get(), 0,
                        sizeof(unsigned long long) * (max_cycle_length + 1)),
             "clear histogram");
  check_cuda(cudaMemset(d_work.get(), 0, sizeof(unsigned long long)), "clear work");
  check_cuda(cudaMemset(d_uncertain.get(), 0, sizeof(unsigned char) * n),
             "clear uncertain");
  check_cuda(cudaMemset(d_flagged.get(), 0, sizeof(unsigned long long)),
             "clear flagged");

  const BlockedGraphView view{graph.vertex_count, d_out_off.get(),
                              d_out_nbr.get(),    d_in_off.get(),
                              d_in_nbr.get(),     d_in_csr.get()};
  const std::size_t shared_bytes = (max_cycle_length + 1) * sizeof(unsigned long long);

  timers.record(timers.kernel_start);
  {
    const ScopedRange range("cuda_blocked_queue_kernel");
    count_roots_blocked_queue_kernel<<<launch.grid_blocks, launch.block_size,
                                       shared_bytes>>>(
        view, static_cast<int>(max_cycle_length), d_work.get(), d_paths.get(),
        d_cursors.get(), d_founds.get(), d_blocked.get(), blocked_words,
        d_b.get(), b_words, d_unblock.get(), unblock_cap, d_hist.get(),
        d_uncertain.get(), d_flagged.get());
    check_cuda(cudaGetLastError(), "count_roots_blocked_queue_kernel launch");
    check_cuda(cudaDeviceSynchronize(),
               "count_roots_blocked_queue_kernel synchronize");
  }
  timers.record(timers.kernel_stop);

  std::vector<unsigned long long> host_hist(max_cycle_length + 1, 0);
  std::vector<unsigned long long> host_flagged(1, 0);
  timers.record(timers.download_start);
  d_hist.copy_to_host(host_hist, "copy histogram");
  d_flagged.copy_to_host(host_flagged, "copy flagged count");
  if (uncertain_out != nullptr) {
    d_uncertain.copy_to_host(*uncertain_out, "copy uncertain");
  }
  timers.record(timers.download_stop);
  timers.record(timers.total_stop);
  if (timings != nullptr) timers.fill(*timings);

  if (flagged_out != nullptr) {
    *flagged_out = static_cast<std::uint64_t>(host_flagged[0]);
  }

  CycleHistogram result;
  for (std::size_t length = 2; length <= max_cycle_length; ++length) {
    if (host_hist[length] == 0) continue;
    result.increment(static_cast<CycleHistogram::Length>(length),
                     static_cast<CycleHistogram::Count>(host_hist[length]));
  }
  return result;
}

}  // namespace detail
}  // namespace cycle_enum::cuda
