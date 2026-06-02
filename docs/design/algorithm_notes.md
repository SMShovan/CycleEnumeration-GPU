# Algorithm Notes

This note summarizes the cycle enumeration algorithms implemented in the project
and how they relate to the SPAA 2022 baseline paper.

## Problem and counting convention

The project counts directed cycles by length. A simple cycle starts and ends at
one vertex and repeats no other vertex; a temporal cycle additionally requires
edge timestamps to strictly increase. To count each simple cycle once, the
search fixes the cycle's smallest vertex as the root and only explores vertices
greater than the root after the root is chosen. For temporal cycles the start
edge timestamp is part of the work item, so one vertex cycle can contribute
several timestamped instances, matching the baseline's model.

## Johnson

Johnson's algorithm is a depth-first search with blocked-set and blocked-list
bookkeeping that avoids re-exploring vertices that cannot currently reach the
root. The sequential implementation uses this pruning directly; the time-window
variant filters each candidate edge by the start window, and the temporal variant
walks timestamped events with strictly increasing timestamps.

## Read-Tarjan

Read-Tarjan enumerates cycles by path extension. Each path-extension search is
more independent than Johnson's recursion, which is why the baseline finds it
easier to parallelize. The project implements static and time-window Read-Tarjan
sequentially and in OpenMP, validated for parity against Johnson and the oracle.

## Temporal techniques

The temporal counters add three ideas from the 2SCENT and baseline lines of
work:

- **Closing-time pruning.** A per-vertex threshold records the earliest time
  from which a vertex can no longer reach the root inside the window; states at
  or beyond that threshold are skipped. The check ignores the path's visited set,
  which keeps it conservative but always sound.
- **Path bundling.** The search carries `(arrival timestamp, multiplicity)`
  bundles per vertex path, merging assignments that reach a vertex at the same
  time, since future reachability depends only on the latest timestamp. This cuts
  branching over timestamp assignments while preserving duplicate multiplicity.
- **Cycle-union preprocessing.** A conservative forward/reverse reachability
  intersection inside the start window drops vertices, and on the CUDA path whole
  start edges, that cannot lie on any valid temporal cycle.

## Relationship to the baseline

The algorithms follow the baseline's choice of Johnson and Read-Tarjan and its
temporal techniques, but the code is rebuilt in a clean, tested structure rather
than copied. The CPU parallelization uses OpenMP instead of TBB tasks, and the
GPU path replaces task stealing with explicit device stacks, a persistent work
queue, and branch splitting, as described in `openmp_strategy.md` and
`cuda_strategy.md`. The brute-force oracle and the sequential validation matrix
in `validation.md` keep every implementation honest.
