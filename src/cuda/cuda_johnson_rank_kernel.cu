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
 * @file cuda_johnson_rank_kernel.cu
 * @brief Rank-based depth-aware blocking cycle counter (the proposed method).
 *
 * One pooled thread claims a root and runs an iterative Johnson search. Instead
 * of a permanent blocked flag plus a stored B-list, each vertex carries a single
 * per-thread value `lvl`:
 *   OPEN (enterable), PERMANENT (never enter), or a truncation RANK r.
 * A blocked neighbor w at child depth cd is skipped iff lvl[w] != OPEN and
 * cd >= lvl[w]. A vertex becomes PERMANENT only if its whole subtree finished
 * with no truncation; otherwise it stores its own depth as the rank, so a later,
 * shallower path (more budget) may re-enter it. Unblocking is done by scanning
 * the CSC in-adjacency (no stored B-list). Simplicity uses an explicit on-path
 * scan. With step_budget == 0 the result is exact for all cycles up to L; with
 * step_budget > 0 a root that exceeds the budget is stopped and flagged, making
 * the histogram a certified lower bound. The iterative logic was validated on CPU
 * against brute force.
 */

namespace cycle_enum::cuda {
namespace detail {
namespace {

constexpr unsigned int kOpen = 0xFFFFFFFFu;  // enterable
constexpr unsigned int kPermanent = 0u;      // never re-enter

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

struct RankGraphView {
  DeviceOffset vertex_count = 0;
  const DeviceOffset* out_off = nullptr;
  const VertexId* out_nbr = nullptr;
  const DeviceOffset* in_off = nullptr;
  const VertexId* in_nbr = nullptr;
};

__global__ void rank_kernel(
    RankGraphView g, int L, unsigned long long* work_counter,
    unsigned int* lvl_all, DeviceOffset lvl_stride, VertexId* V_all,
    DeviceOffset* Cur_all, unsigned char* Fnd_all, unsigned char* Trn_all,
    DeviceOffset stack_stride, VertexId* U_all, DeviceOffset u_cap,
    unsigned long long step_budget, unsigned long long* histogram,
    unsigned char* uncertain, unsigned long long* flagged_count) {
  extern __shared__ unsigned long long bh[];
  for (int i = threadIdx.x; i <= L; i += blockDim.x) bh[i] = 0ULL;
  __syncthreads();

  const DeviceOffset tid =
      static_cast<DeviceOffset>(blockIdx.x) * blockDim.x + threadIdx.x;
  unsigned int* lvl = lvl_all + tid * lvl_stride;
  VertexId* V = V_all + tid * stack_stride;
  DeviceOffset* Cur = Cur_all + tid * stack_stride;
  unsigned char* Fnd = Fnd_all + tid * stack_stride;
  unsigned char* Trn = Trn_all + tid * stack_stride;
  VertexId* U = U_all + tid * u_cap;
  const DeviceOffset n = g.vertex_count;

  while (true) {
    const DeviceOffset root = atomicAdd(work_counter, 1ULL);
    if (root >= n) break;

    for (DeviceOffset i = 0; i < n; ++i) lvl[i] = kOpen;  // per-root reset
    unsigned long long steps = 0;
    bool budget_hit = false;

    int depth = 1;
    V[0] = static_cast<VertexId>(root);
    Cur[0] = g.out_off[root];
    Fnd[0] = 0;
    Trn[0] = 0;

    while (depth >= 1) {
      if (step_budget > 0 && steps >= step_budget) {
        budget_hit = true;
        break;
      }
      const int idx = depth - 1;
      const VertexId v = V[idx];
      const DeviceOffset vend = g.out_off[v + 1];

      if (Cur[idx] < vend) {
        const VertexId w = g.out_nbr[Cur[idx]];
        ++Cur[idx];
        ++steps;
        if (w == static_cast<VertexId>(root) && depth >= 2) {
          atomicAdd(&bh[depth], 1ULL);
          Fnd[idx] = 1;
          continue;
        }
        if (w <= static_cast<VertexId>(root)) continue;
        bool on_path = false;
        for (int i = 0; i < depth; ++i) {
          if (V[i] == w) { on_path = true; break; }
        }
        if (on_path) continue;
        const int cd = depth + 1;
        if (cd > L) { Trn[idx] = 1; continue; }
        const unsigned int lw = lvl[w];
        if (lw != kOpen && static_cast<unsigned int>(cd) >= lw) {
          if (lw != kPermanent) Trn[idx] = 1;  // skipped a truncated block
          continue;
        }
        const int nidx = depth;  // == cd - 1
        V[nidx] = w;
        Cur[nidx] = g.out_off[w];
        Fnd[nidx] = 0;
        Trn[nidx] = 0;
        depth = cd;
      } else {
        const int vd = depth;
        const unsigned char fv = Fnd[idx];
        const unsigned char tv = Trn[idx];
        const VertexId vv = V[idx];
        --depth;
        if (depth >= 1) {
          const int pidx = depth - 1;
          Fnd[pidx] |= fv;
          Trn[pidx] |= tv;
        }
        if (fv != 0) {
          // unblock(vv): open vv and cascade to blocked predecessors via CSC.
          DeviceOffset top = 0;
          U[top++] = vv;
          lvl[vv] = kOpen;
          while (top > 0) {
            const VertexId x = U[--top];
            const DeviceOffset e0 = g.in_off[x];
            const DeviceOffset e1 = g.in_off[x + 1];
            for (DeviceOffset k = e0; k < e1; ++k) {
              const VertexId p = g.in_nbr[k];
              if (lvl[p] != kOpen) {
                lvl[p] = kOpen;
                if (top < u_cap) {
                  U[top++] = p;
                } else {
                  // scratch overflow: open everything (still exact, less pruning)
                  for (DeviceOffset i = 0; i < n; ++i) lvl[i] = kOpen;
                  top = 0;
                  break;
                }
              }
            }
          }
        } else if (depth >= 1) {
          lvl[vv] = (tv != 0) ? static_cast<unsigned int>(vd) : kPermanent;
        }
      }
    }

    uncertain[root] = budget_hit ? 1 : 0;
    if (budget_hit) atomicAdd(flagged_count, 1ULL);
  }

  __syncthreads();
  for (int i = threadIdx.x; i <= L; i += blockDim.x) {
    const unsigned long long c = bh[i];
    if (c != 0ULL) atomicAdd(&histogram[i], c);
  }
}

}  // namespace

CycleHistogram count_simple_cycles_johnson_rank_queue_device(
    const CudaGraphData& graph, const int device_id,
    const std::size_t max_cycle_length, CudaTimingsMs* const timings,
    const std::uint64_t step_budget, std::vector<unsigned char>* const uncertain_out,
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
  const DeviceOffset lvl_stride = graph.vertex_count;
  const DeviceOffset stack_stride = L;  // max path index is L-1
  const DeviceOffset u_cap =
      (graph.vertex_count + 1 <= 4096) ? (graph.vertex_count + 1) : 4096;

  // CSC in-neighbor sources, extracted from the AoS incoming edges (no B-list,
  // no cross-map needed; only the predecessor list is used for unblock).
  std::vector<VertexId> in_neighbors;
  in_neighbors.reserve(graph.incoming_edges.size());
  for (const CudaAdjacencyEntry& entry : graph.incoming_edges) {
    in_neighbors.push_back(entry.vertex);
  }

  // Clamp the resident grid to available memory.
  const std::size_t per_thread =
      static_cast<std::size_t>(lvl_stride) * sizeof(unsigned int) +
      static_cast<std::size_t>(stack_stride) *
          (sizeof(VertexId) + sizeof(DeviceOffset) + 2 * sizeof(unsigned char)) +
      static_cast<std::size_t>(u_cap) * sizeof(VertexId);
  std::size_t free_bytes = 0, total_bytes = 0;
  check_cuda(cudaMemGetInfo(&free_bytes, &total_bytes), "cudaMemGetInfo");
  const std::size_t graph_bytes =
      (graph.outgoing_offsets.size() + graph.incoming_offsets.size()) * sizeof(DeviceOffset) +
      (graph.outgoing_neighbors.size() + in_neighbors.size()) * sizeof(VertexId) +
      (max_cycle_length + 1) * sizeof(unsigned long long) +
      n * sizeof(unsigned char);
  const std::size_t budget_bytes =
      free_bytes > graph_bytes ? static_cast<std::size_t>((free_bytes - graph_bytes) * 0.8)
                               : 0;
  const DeviceOffset planned_threads =
      static_cast<DeviceOffset>(launch.grid_blocks) * launch.block_size;
  if (per_thread > 0) {
    const DeviceOffset affordable =
        static_cast<DeviceOffset>(budget_bytes / per_thread);
    if (affordable < planned_threads) {
      const DeviceOffset blocks = affordable / launch.block_size;
      if (blocks == 0) {
        throw std::runtime_error(
            "insufficient GPU memory for the rank backend; reduce "
            "--max-cycle-length or run on a smaller graph");
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
  DeviceBuffer<VertexId> d_in_nbr(in_neighbors.size());
  d_out_off.copy_from_host(graph.outgoing_offsets, "copy out offsets");
  d_out_nbr.copy_from_host(graph.outgoing_neighbors, "copy out neighbors");
  d_in_off.copy_from_host(graph.incoming_offsets, "copy in offsets");
  d_in_nbr.copy_from_host(in_neighbors, "copy in neighbors");
  timers.record(timers.upload_stop);

  DeviceBuffer<unsigned int> d_lvl(static_cast<std::size_t>(total_threads * lvl_stride));
  DeviceBuffer<VertexId> d_V(static_cast<std::size_t>(total_threads * stack_stride));
  DeviceBuffer<DeviceOffset> d_Cur(static_cast<std::size_t>(total_threads * stack_stride));
  DeviceBuffer<unsigned char> d_Fnd(static_cast<std::size_t>(total_threads * stack_stride));
  DeviceBuffer<unsigned char> d_Trn(static_cast<std::size_t>(total_threads * stack_stride));
  DeviceBuffer<VertexId> d_U(static_cast<std::size_t>(total_threads * u_cap));
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

  const RankGraphView view{graph.vertex_count, d_out_off.get(), d_out_nbr.get(),
                           d_in_off.get(), d_in_nbr.get()};
  const std::size_t shared_bytes = (max_cycle_length + 1) * sizeof(unsigned long long);

  timers.record(timers.kernel_start);
  {
    const ScopedRange range("cuda_rank_kernel");
    rank_kernel<<<launch.grid_blocks, launch.block_size, shared_bytes>>>(
        view, static_cast<int>(max_cycle_length), d_work.get(), d_lvl.get(),
        lvl_stride, d_V.get(), d_Cur.get(), d_Fnd.get(), d_Trn.get(),
        stack_stride, d_U.get(), u_cap, step_budget, d_hist.get(),
        d_uncertain.get(), d_flagged.get());
    check_cuda(cudaGetLastError(), "rank_kernel launch");
    check_cuda(cudaDeviceSynchronize(), "rank_kernel synchronize");
  }
  timers.record(timers.kernel_stop);

  std::vector<unsigned long long> host_hist(max_cycle_length + 1, 0);
  std::vector<unsigned long long> host_flagged(1, 0);
  timers.record(timers.download_start);
  d_hist.copy_to_host(host_hist, "copy histogram");
  d_flagged.copy_to_host(host_flagged, "copy flagged");
  if (uncertain_out != nullptr) {
    d_uncertain.copy_to_host(*uncertain_out, "copy uncertain");
  }
  timers.record(timers.download_stop);
  timers.record(timers.total_stop);
  if (timings != nullptr) timers.fill(*timings);
  if (flagged_out != nullptr) *flagged_out = static_cast<std::uint64_t>(host_flagged[0]);

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
