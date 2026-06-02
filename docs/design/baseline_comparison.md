# Baseline Comparison Protocol

This note describes how to compare CycleEnumeration-GPU against the SPAA 2022
TBB baseline (`parallel-cycle-enumeration`). The comparison covers both
correctness (matching cycle histograms) and performance (runtime), and is kept
outside the normal build so this project never depends on TBB or MPI.

## Building the baseline

Build the reference project separately, following its own instructions, to
produce its `cycle` executable. It is not a submodule and is not required to
build or test CycleEnumeration-GPU; it is only needed for comparison runs.

## Input compatibility

Both programs read the same `source target timestamp` edge-list format, so a
dataset is used unchanged. Use the same time window for both. The harness passes
the baseline `-tws` so its time window is always interpreted in seconds, matching
the project CLI's `--time-window`.

## Algorithm correspondence

The baseline selects algorithms by numeric id (see `benchmarks/README.md`). The
ids relevant to this project are:

| Project run | Baseline id |
| --- | --- |
| time-window Johnson | 0 (coarse) or 1 (fine) |
| time-window Read-Tarjan | 2 (coarse) or 3 (fine) |
| temporal Johnson | 4 (coarse) or 5 (fine) |
| temporal Read-Tarjan | 6 (coarse) or 7 (fine) |

Correctness must match across the coarse and fine baseline variants since they
compute the same cycles; performance is compared against the fine-grained
variants, which are the baseline's scalable path.

## Correctness comparison

`scripts/compare_reference.sh` runs the project CLI and the baseline `cycle`
executable on the same input and mode and diffs their cycle histograms. The two
histograms must be identical. A mismatch is a correctness defect and blocks any
performance claim until resolved. The script normalizes both outputs to
`length count` pairs so formatting differences do not cause false mismatches.

## Performance comparison

After histograms match, use `benchmarks/run_cycle_benchmarks.py` with
`--tbb-exe` to record project and baseline runtimes in one CSV, then
`scripts/collect_results.py` and `scripts/plot_results.py` to summarize. Compare
the project's OpenMP and CUDA backends against the fine-grained baseline at the
same thread count, and report speedup relative to the baseline alongside the
absolute times.

## Recording

For every comparison record the graph name and size, the build configurations of
both programs, the CPU and GPU models, the compiler and CUDA versions, the thread
counts, and the Slurm allocation. The CSV command column preserves the exact
arguments; the surrounding machine metadata lives in the experiment log so a
published speedup can be reproduced.
