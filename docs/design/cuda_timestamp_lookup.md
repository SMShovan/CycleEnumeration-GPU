# CUDA Timestamp Lookup

This note covers how the CUDA temporal kernels locate usable edge timestamps and
the optimization that reduces redundant timestamp work.

## Baseline: device binary search

The naive temporal and time-window kernels keep edge timestamps as a sorted flat
array and binary search it in the inner loop, through the device helpers in
`cuda_timestamp.hpp` (`timestamps_after`, `has_timestamp_in_window`). Binary
search is correct and matches the CPU boundary convention, but it has two costs
in the hot loop: each search is several dependent global-memory loads, and an
edge with duplicate timestamp values produces one device frame per duplicate
even though all duplicates lead to identical future extensions.

## Optimization: distinct-timestamp grouping

`build_outgoing_timestamp_groups` precomputes, per outgoing edge slot, the
distinct timestamp values with their multiplicities. Because the timestamp list
is already sorted, grouping is a single linear pass that merges adjacent equal
values. The result is a CSR-style layout (`edge_group_offsets`,
`group_values`, `group_counts`) indexed exactly like the structure-of-arrays
neighbor array.

The temporal kernel then iterates distinct timestamp values instead of raw
events. When a cycle closes, it adds the accumulated group multiplicity to the
histogram rather than `1`. This preserves duplicate-event multiplicity, matching
the sequential temporal counter, while cutting the number of timestamp branches
on edges that have repeated timestamps. It is the host half of the hybrid
bundling decision recorded in `cuda_temporal_bundling.md`.

## Caching valid ranges

For a fixed window the kernel still needs the sub-range of an edge's groups that
is strictly after the current arrival time and within the window. The group
arrays keep that sub-range a small binary search over distinct values rather than
over raw events, so duplicate-heavy edges no longer inflate the search. Fully
precomputing per-edge valid ranges is start-time dependent and is only worthwhile
when one window is reused across many start events; it is left as a measured
option because it trades memory for fewer searches.

## Validation and measurement

The grouping is host-only and unit tested: counts sum back to each edge's
original timestamp count, distinct values stay strictly increasing per edge, and
the offset layout is consistent. The effect on kernel time and on the number of
device frames is measured on the H100 cluster against the binary-search baseline,
with at least one duplicate-timestamp fixture so the multiplicity path is
exercised.
