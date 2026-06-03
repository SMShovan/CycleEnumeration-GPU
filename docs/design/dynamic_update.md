# Dynamic Histogram Update

This note describes the incremental update of a simple-cycle histogram under a
batch of edge changes, without recomputing the whole graph. The first
implementation targets **static simple cycles** only; time-window and temporal
updates are deferred.

## Problem

Given an initial directed graph `G0`, its cycle histogram `H0` (computed by the
existing recompute path), and a batch `B` of edge **deletions** and
**insertions**, produce the histogram of the resulting graph `G_final` by
adjusting `H0`, touching only the cycles affected by the changed edges.

## Locality principle

Inserting or deleting an edge `(u, v)` can only create or destroy simple cycles
that pass through `(u, v)`. Every cycle that avoids all changed edges is
unaffected. So the update enumerates only cycles through the changed edges and
adjusts their length buckets.

The cycles through a directed edge `(u, v)` are exactly the edge plus every
simple path from `v` back to `u`. A path of `k` edges yields a cycle of length
`k + 1`. This is an edge-anchored depth-first search, bounded by the maximum
cycle length `L`.

## Delete-then-insert with edge-id ownership

A batch is applied in two ordered phases, each enumerated against a frozen graph
snapshot so the phases can run in parallel internally:

1. **Delete phase** — enumerate against `G0` (all deleted edges still present).
   For each deleted edge, enumerate the cycles through it and **decrement** their
   length buckets.
2. **Insert phase** — enumerate against `G_final` (all deletions applied, all
   insertions present). For each inserted edge, enumerate the cycles through it
   and **increment** their length buckets.

Within a phase, a cycle may pass through several changed edges. To count it once
without sequential coordination, each cycle is **owned by the smallest-id
changed edge it contains** (ids are assigned per phase). A changed edge only
adjusts cycles it owns. Equivalently, while searching from changed edge `x`, the
path is forbidden from traversing any same-phase changed edge with a smaller id;
this both enforces single ownership and prunes redundant work.

### Why this is exact

Classify each cycle by the changed edges it uses:

| Cycle uses | Net change | Mechanism |
| --- | --- | --- |
| no changed edge | 0 | never enumerated |
| a deleted edge, no inserted edge | -1 | decremented once by its min-id deleted owner |
| an inserted edge, no deleted edge | +1 | incremented once by its min-id inserted owner |
| both a deleted and an inserted edge | 0 | absent from `G0` (insert missing) and from `G_final` (delete missing); enumerated in neither phase |

The deleted-and-inserted "phantom" never appears, so the result equals the
histogram of `G_final`. This equality is the golden correctness gate:
`update(H0, B) == recompute(G_final)`.

## Counting convention

The recompute path counts each distinct directed simple cycle once, rooted at its
smallest vertex. The update counts each distinct directed simple cycle through a
changed edge once, owned by its min-id changed edge. Both count the same objects
(distinct directed simple cycles by length), so the histograms are comparable.
The update does **not** use smallest-vertex rooting; ownership is by edge, not by
vertex.

## Arithmetic

`CycleHistogram` counts are unsigned and have no decrement primitive. The update
accumulates a signed per-length delta map across both phases, then forms the
result as `H0[length] + delta[length]` for each length, so a bucket never
transiently underflows.

## Parallelism (later backends)

Both phases are read-only over a frozen snapshot, so each phase parallelizes over
its changed edges with no coordination beyond atomic histogram updates. A heavy
changed edge's search is balanced internally with the same persistent work queue
and branch splitting used by the static counters. Only the phase boundary (apply
all deletions, then enumerate insertions on `G_final`) is a synchronization
point.

## Performance notes

- **Visited buffer reuse.** The edge-anchored search restores every vertex it
  marks on backtrack, so a per-thread visited buffer stays all-zero between
  changed edges once the anchored endpoints are cleared. Reusing it avoids an
  `O(vertex_count)` allocation and reset per changed edge.
- **Incremental graph mutation.** Building the post-batch graph only rebuilds and
  re-sorts the source vertices the batch touches; every other vertex's sorted
  neighbor range is copied directly. A production streaming system would instead
  maintain the post-batch adjacency incrementally in `O(batch)` rather than
  rebuilding it; the benchmark recompute baseline pays the same construction, so
  the comparison stays fair.
- **Ownership as pruning.** The ownership check is a hash or binary-search lookup
  in the inner loop; refusing to cross a smaller-id changed edge both enforces
  single attribution and cuts redundant traversal of cycles owned elsewhere.
- **Open follow-ups.** The per-edge parallelism (OpenMP and one CUDA thread per
  changed edge) under-fills the device for batches with few but heavy edges; the
  persistent work queue and branch splitting would balance that and are the
  natural next optimization.

## Validation

Correctness is established entirely on the host: for many random graphs, random
batches, and seeds, assert `update(H0, B) == recompute(G_final)` for static
simple cycles, including empty batches, all-delete and all-insert batches,
delete-then-reinsert of the same edge, overlapping changed edges, two-cycles, and
the maximum-length boundary. The sequential reference is the oracle for the
OpenMP and CUDA updates; the recompute path is the oracle for the sequential
update.
