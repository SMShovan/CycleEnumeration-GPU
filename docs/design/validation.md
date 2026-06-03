# Validation Strategy

The project keeps a staged correctness chain so optimized implementations never
stand alone:

```text
tiny brute-force oracle
  -> sequential Johnson / Read-Tarjan
  -> OpenMP implementations
  -> CUDA implementations
```

## Current Sequential Coverage

The current test suite validates these shared components:

- Temporal graph parsing, duplicate edge grouping, and compact vertex ordering.
- CSR/CSC graph views and timestamp-range helpers.
- Histogram formatting, merge behavior, and overflow checks.
- Static simple-cycle counting with brute force, Johnson, and Read-Tarjan.
- Simple time-window counting with brute force, Johnson, and Read-Tarjan.
- Temporal cycle counting with brute force, temporal Johnson, and temporal
  Read-Tarjan.
- Temporal Johnson closing-time pruning and timestamp path bundling.
- Sequential cycle-union preprocessing as a safe pruning superset.
- Sequential CLI output for static and simple time-window modes.

The `SequentialParityTest` integration suite compares algorithms by semantic
mode and acts as the CPU reference matrix for later OpenMP and CUDA work.

## CUDA Validation Policy

CUDA cannot be executed on the local Mac development machine, so CUDA tests will
be structured to compile only when CUDA is enabled and to skip cleanly when no
device is present. On an NVIDIA H100 cluster, CUDA results should be checked
against the sequential matrix on small and medium fixtures before collecting
performance numbers.

## Benchmark Correctness Gate

Benchmark scripts should compare histograms before reporting speedups. A faster
implementation with a mismatched histogram is treated as incorrect, not as a
performance result.

## Final Validation Summary

### Host-side test coverage

Beyond the sequential and OpenMP parity suites, the host-side building blocks of
the CUDA backend are unit tested on the development machine:

- Graph packing into the structure-of-arrays device layout.
- Start-event generation and the temporal cycle-union prefilter, checked against
  an independent temporal-reachability reference so it never drops a cyclic start
  edge.
- Branch-split decomposition, checked to recombine exactly to the sequential
  Johnson count across cutoff depths and a bounded-length case.
- Timestamp grouping (count preservation, per-edge ordering, offset
  consistency).
- Bitset visited-set operations across word boundaries and the mode selector.
- The work-queue launch planner and the environment-driven launch tuning reader.
- The NVTX scoped-range no-op path and the CLI scheduler option.

The OpenMP task experiment is additionally validated under a real OpenMP runtime
for histogram parity at several thread counts and cutoff depths.

### GPU validation (cluster)

Kernel compilation, GPU correctness parity against the sequential matrix, and
timing are performed on the NVIDIA H100 cluster, since the development machine has
no CUDA device. The naive and work-queue static paths, the time-window path, and
the temporal path are each checked against the sequential counter on small and
medium fixtures before any performance number is recorded.

### Benchmark environment to record

For every reported result, record the graph name and size (vertices, directed
edges, timestamped events), both build configurations, the CPU and GPU models,
the compiler and CUDA versions, the thread counts and CUDA launch configuration,
and the Slurm allocation. The benchmark CSV preserves the exact command line; the
machine metadata lives in the experiment log.

### Dynamic update correctness gate

The incremental update is validated entirely on the host against full
recomputation, which is the same oracle used for the static counters:

- A randomized property test asserts that the sequential update equals a full
  recomputation across 200 random graph-and-batch trials, plus empty, all-delete,
  and all-insert batches.
- The OpenMP update is checked against both the sequential update and a full
  recomputation at several thread counts.
- The CUDA update's host dispatch and unavailable-backend path are tested
  locally; its device parity against full recomputation is validated on the
  cluster and skips cleanly without a device.
- The enumeration primitive, batch generator, edge-change types, and directed
  graph mutation have their own unit tests.

Discovering this gate also surfaced and fixed a latent bug in the length-bounded
baseline Johnson counter (see the changelog); a randomized regression test now
compares bounded Johnson against the brute-force oracle.

### Known limitations

- The incremental update covers static simple cycles only; time-window and
  temporal updates are not yet implemented.
- CUDA modes use a configurable maximum cycle length to bound device stacks; they
  do not yet support unbounded depth.
- The work-queue and branch-split optimizations currently target the static
  path; the time-window and temporal paths use the naive kernels with the host
  prefilter.
- Cycle listing is out of scope; the project reports histograms only.
- GPU performance numbers are produced on the cluster and are not part of the
  local test suite.
