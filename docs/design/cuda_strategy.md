# CUDA Strategy

This note summarizes how the CUDA backend is structured and how it differs from
the paper's TBB task implementation. It also records which CUDA mode is the
default and which is kept for debugging.

## Why not a direct TBB port

The baseline uses a task-stealing recursion tree on the CPU. That maps poorly to
the GPU, where threads need compact state, predictable memory access, and many
similar work items. The CUDA backend therefore rebuilds the search around
GPU-friendly building blocks rather than translating tasks one to one:

- **Flat, split graph layout.** The hot outgoing adjacency is stored as a
  structure of arrays so warp loads coalesce and the static path never touches
  timestamp data (`cuda_memory_layout.md`).
- **Explicit device stacks.** The depth-first search uses an explicit
  per-thread stack instead of recursion, with the depth bounded by
  `max_cycle_length`.
- **Host-built work items.** Start events and, for temporal mode, the
  cycle-union prefilter are computed on the host and streamed to the device
  (`cuda_cycle_union.md`).
- **Dynamic scheduling.** A persistent work-queue kernel lets resident blocks
  claim work from a global counter, and branch splitting decomposes heavy roots
  into many small work items, both attacking the load imbalance that limits a
  one-thread-per-work-item kernel (`cuda_work_splitting.md`).
- **Low-contention aggregation.** Cycle counts accumulate in a per-block shared
  histogram and flush once to global memory.

## Naive versus optimized modes

Two static CUDA paths exist:

- **Naive** (`count_simple_cycles_johnson`): one root per thread. It is the
  simplest correct GPU baseline and stays available for debugging and parity
  checks.
- **Work queue** (`count_simple_cycles_johnson_work_queue`): persistent blocks
  claim roots dynamically. It is the default static CUDA path because it keeps
  the device busy on skewed graphs.

The CLI selects between them with `--cuda-scheduler <naive|work-queue>`, default
`work-queue`. The time-window and temporal CUDA paths currently use the naive
per-start-event kernels with the host prefilter; their work-queue and
branch-split integration reuses the same building blocks and is validated on the
cluster.

## Validation boundary

Host-side pieces — graph packing, work-item generation, cycle-union prefilter,
branch-split decomposition, visited-set primitives, timestamp grouping, launch
planning and tuning — are unit tested on the development machine. Kernel
compilation, GPU correctness parity against the sequential oracle, and timing
are performed on the H100 cluster, as recorded in `validation.md`.
