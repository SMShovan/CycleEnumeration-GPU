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
