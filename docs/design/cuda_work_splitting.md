# CUDA Branch Splitting

This note describes how a heavy cycle search is split into many balanced work
items for the CUDA backend, and why the split is exact.

## Problem

The naive and work-queue kernels treat one root vertex as one unit of work. On
skewed graphs a few high-degree roots own most of the search tree, so even with
a work queue a single block can be stuck on one enormous root while the rest of
the device drains. Branch splitting breaks that one heavy unit into many small
units that the work queue can spread out.

## Host decomposition

`split_static_search` walks every simple path prefix up to `cutoff_depth`
vertices, rooted at the smallest vertex of each prefix. It produces:

- `closed`: a histogram of every cycle whose length is strictly less than the
  cutoff. These cycles are fully discovered during the host pre-pass.
- `items`: one `SplitWorkItem` per path prefix of exactly `cutoff_depth`
  vertices that still has an unexplored continuation. The prefix records the
  cycle root (`prefix.front()`) and the vertex to resume from (`prefix.back()`),
  and its vertices form the already-visited set for the continuation.

The device resumes a depth-first search from each work item and counts every
closure it finds, which are exactly the cycles of length `>= cutoff_depth`.

## Exactness

The boundary rule is deliberately clean so there is no double counting: the host
owns cycles of length `< cutoff_depth`, and the device owns cycles of length
`>= cutoff_depth`. The host stops one level before the cutoff and never counts a
closure at the cutoff length, so the device's first-level closures are not
duplicated. The smallest-root duplicate-avoidance rule is unchanged, since every
prefix keeps its root fixed and only explores vertices greater than the root.

Combining `closed` with the device continuations therefore reproduces the
undecomposed count exactly. This is verified in `cuda_branch_split_test.cpp`,
which models the device continuation on the host and asserts that the
recombined histogram equals the sequential Johnson count for several cutoff
depths, including a bounded `max_cycle_length` case.

## Choosing the cutoff

A larger `cutoff_depth` produces more and smaller work items, exposing more
parallelism at the cost of a longer host pre-pass and more prefix storage. A
cutoff of 2 already turns each root into one work item per outgoing edge, which
is usually enough to break up the dominant roots; deeper cutoffs are reserved
for graphs with a very small number of extremely heavy roots. The cutoff is a
tuning parameter swept alongside block and queue sizes in the launch tuning
step.

## Device integration

The work items feed the same persistent work queue used for root-level
scheduling (`cuda_work_queue.hpp`): instead of claiming a root, a block claims a
prefix and resumes the search from `prefix.back()` with the prefix marked
visited. The host decomposition and its exactness guarantee are validated
locally; the kernel that consumes the prefixes is validated on the H100 cluster
against the sequential counter.
