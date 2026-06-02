# OpenMP Read-Tarjan Task Experiment

This note documents the fine-grained OpenMP task variant of the Read-Tarjan
static cycle counter. It is an experiment for comparison, not the primary CPU
baseline. The coarse-grained Read-Tarjan counter remains the reference CPU
implementation because it is simpler and has predictable memory behavior.

## Motivation

Coarse-grained parallelism assigns one root vertex to one loop iteration. This
is work efficient but suffers load imbalance when a few roots own most of the
search tree, which is common in real temporal graphs with skewed degree
distributions. The baseline paper addresses this with fine-grained tasks and a
copy-on-steal strategy for path and blocked state. This experiment reproduces
the same idea with OpenMP tasks so the project can measure whether finer
granularity helps the CPU baseline before the same lesson is applied on the GPU
work queue (Phase 8.3).

## Design

- One OpenMP task is created per root vertex from a single producer thread
  inside an `omp single nowait` region.
- While a path prefix is shorter than `task_cutoff_depth`, each child branch is
  spawned as a new task. At or beyond the cutoff, the remaining subtree is
  searched serially inside the running task.
- Every spawned task receives private copies of the current path and visited
  state through `firstprivate`. This is the copy-on-steal step: concurrent tasks
  never share mutable depth-first state, so no locking is needed inside the
  search.
- Each task accumulates cycle counts into the histogram owned by its executing
  thread (`omp_get_thread_num()`); these per-thread histograms are merged after
  the parallel region.

## Cutoff trade-off

`task_cutoff_depth` controls granularity:

- `1` creates one task per root, equivalent to coarse-grained scheduling but
  routed through the task scheduler.
- `2` (the default) additionally splits each root's immediate children into
  tasks, which exposes more parallelism for high-degree roots.
- Higher values expose still more parallelism but increase task-creation
  overhead and the amount of path and visited state copied per task.

The state copying cost grows with both the cutoff depth and the vertex count,
because the visited array is copied per spawned task. The experiment is intended
for small and medium graphs where exposing parallelism matters more than the
copy overhead.

## Correctness

The task variant produces exactly the same histogram as the sequential and
coarse-grained Read-Tarjan counters; only the scheduling differs. The duplicate
avoidance rule is unchanged: a cycle is counted once, rooted at its smallest
vertex, and only vertices greater than the root are explored after the root is
chosen. Parity is checked in `openmp_task_parity_test.cpp` against both the
sequential Read-Tarjan counter and the coarse-grained OpenMP counter across the
shared fixtures, for one and several requested threads and for several cutoff
depths.

## When OpenMP is unavailable

Without an OpenMP runtime, or for a single-thread request, the function runs the
same serial depth-first search used by the fallback path of the coarse-grained
counter, so results and tests remain valid on the local macOS build.
