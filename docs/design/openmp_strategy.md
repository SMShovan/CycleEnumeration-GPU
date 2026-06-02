# OpenMP Strategy

This note describes the OpenMP CPU parallelization and why it is the project's
CPU baseline for comparison against the GPU.

## Goals

The OpenMP backend does not need extreme optimization. It needs to be correct,
deterministic for a given graph, measurable, and reasonably scalable, so it gives
a fair CPU reference point for the CUDA results and a bridge to the TBB baseline.

## Coarse-grained parallelism

The primary OpenMP counters parallelize over independent start work items: root
vertices for static and time-window cycles, and start-edge timestamp events for
temporal cycles. Each worker keeps its depth-first state — path, visited set,
closing-time caches, and timestamp bundles — entirely thread-local, so no
synchronization is needed inside the search. Cycle counts accumulate into a
per-thread histogram, and the histograms are merged after the parallel region.
Work is distributed with dynamic scheduling so that a few heavy roots do not
leave most threads idle.

This coarse-grained scheme is work efficient: it performs the same search as the
sequential counter, just split across threads. Its weakness is load imbalance
when a small number of roots own most of the work, which is exactly the case
dynamic scheduling and, on the GPU, the work queue and branch splitting address.

## Fine-grained task experiment

A separate Read-Tarjan task variant spawns an OpenMP task per path prefix up to a
cutoff depth and copies the path and visited state into each task, mirroring the
copy-on-steal idea from the baseline. It is a measurable experiment, not the
primary baseline, kept to compare finer granularity against the coarse scheme
(see `openmp_tasks.md`). It produces an identical histogram; only scheduling
differs.

## Availability and fallback

OpenMP is optional. Without an OpenMP runtime, or for a single-thread request,
every OpenMP counter runs the same serial depth-first search, so results and
tests remain valid on a machine without OpenMP. Thread-count requests are
validated, and a multi-thread request without an OpenMP runtime fails with a
clear message rather than silently running serially.

## Relationship to the GPU path

The OpenMP backend establishes the CPU scaling curve that the CUDA backend is
measured against. The load-imbalance lesson it surfaces — that coarse, per-root
parallelism stalls on skewed graphs — directly motivates the GPU persistent work
queue and branch splitting described in `cuda_strategy.md`.
