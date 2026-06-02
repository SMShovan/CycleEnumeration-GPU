# CUDA Temporal Path-Bundling Investigation

This note decides how, and whether, the sequential temporal path-bundling
optimization should be carried over to the CUDA temporal kernel. It is the
Phase 7.4 investigation; the implementation lands later in the CUDA optimization
phase.

## Background

The sequential temporal Johnson search bundles timestamp assignments. While
walking a fixed vertex path it carries, for the current vertex, a set of
`(arrival_timestamp, multiplicity)` entries and merges assignments that reach a
vertex with the same timestamp, because future temporal reachability depends
only on the latest timestamp. This reduces recursive branching over timestamp
assignments while preserving duplicate-event multiplicity (see
`temporal_johnson.md`).

The naive CUDA temporal kernel does the opposite: it assigns one start event to
one thread and explicitly branches over every later timestamp through per-thread
DFS frames (`active_edges`, `timestamp_cursors`, `arrival_timestamps`). It is
correct, but every duplicate timestamp and every distinct arrival time is a
separate device frame, which inflates per-thread state and warp divergence.

## Options considered

1. **Full on-GPU bundling.** Each thread keeps a variable-length bundle of
   `(timestamp, multiplicity)` entries per DFS frame and merges them on the
   device. This is the most faithful port, but bundle width is data dependent
   and unbounded, so it forces dynamic per-thread storage, raises register and
   local-memory pressure, lowers occupancy, and adds divergent merge loops. On
   the GPU execution model the bundling overhead can erase the branching it
   saves. High correctness complexity.

2. **CPU/host-only bundling.** Precompute bundles on the host and stream a
   reduced work set to the device. The merge step that makes bundling powerful
   is path dependent (it merges assignments that converge at a vertex *along a
   specific path*), so it cannot be fully precomputed without enumerating paths,
   which defeats the purpose. Only the trivial, path-independent part —
   collapsing exact-duplicate timestamps on a single edge — can be precomputed
   cheaply.

3. **Hybrid: host timestamp grouping plus a device multiplicity scalar
   (chosen).** The host collapses each directed edge's timestamp list into
   `(distinct_timestamp, count)` groups. The kernel iterates distinct timestamp
   values instead of raw events, and carries a single accumulated multiplicity
   integer down the DFS path. When a cycle closes, it adds the accumulated
   multiplicity to the histogram instead of `1`. This captures exactly the
   duplicate-timestamp multiplicity that sequential bundling preserves, with
   O(1) extra per-frame state, bounded memory, and no divergent merge loops.

## Decision

Adopt option 3, the hybrid. It removes the largest and simplest source of
redundant device branching — duplicate timestamps on the same edge — at the cost
of one extra integer per DFS frame and a cheap host pre-pass, neither of which
fights the GPU execution model.

Full path-dependent bundling (option 1), which also merges assignments that
arrive at the same vertex with the same timestamp from *different* predecessors,
is deferred. It only pays off on graphs where many distinct paths reconverge in
time, and its unbounded per-thread state conflicts with occupancy goals. It
should be revisited only if profiling on real temporal datasets shows that
reconvergent timestamp assignments, rather than load imbalance, dominate the
temporal kernel runtime. Load imbalance is addressed first by the persistent
work queue and branch splitting in Phase 8.

## Correctness contract

The hybrid does not change the histogram contract. The naive kernel already
counts each timestamped cycle instance once by treating every event separately;
grouping duplicate timestamps and multiplying by their count produces the same
totals because all events with one timestamp value yield identical future
extensions. The change is validated on the cluster against the sequential
temporal Johnson counter and the temporal brute-force oracle, with at least one
fixture containing duplicate timestamps on a cycle edge so the multiplicity path
is exercised.

## Implementation pointer

The host grouping is a natural extension of the start-event and timestamp
helpers and is implemented together with the timestamp-lookup optimization in
Phase 8.6. The device multiplicity scalar is added to the temporal kernel frames
at the same time, behind the existing CUDA build guard.
