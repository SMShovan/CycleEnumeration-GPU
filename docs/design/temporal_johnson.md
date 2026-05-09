# Sequential Temporal Johnson Notes

The first temporal Johnson implementation is a correctness-first DFS over
timestamped edge events. Each work item is a start edge event `(root, next,
start_time)`. The search then extends only through edge events whose timestamp
is strictly greater than the previous timestamp and no greater than
`start_time + window_width`.

The implementation counts timestamped cycle instances. If one vertex cycle has
multiple valid timestamp assignments, each assignment contributes to the
histogram.

## Closing-Time Pruning

The initial pruning rule is conservative. For a fixed root and fixed window
end, the search maintains a failure threshold per vertex:

```text
closing_after[v] = earliest timestamp t where v cannot reach root after t
```

If the DFS later reaches the same vertex with `previous_timestamp >=
closing_after[v]`, that state is skipped. This is safe because temporal
reachability is monotonic: starting later leaves a subset of the edge events
available.

The reachability check intentionally ignores the current simple-path visited
set. That makes the pruning weaker than a fully path-aware scheme, but it keeps
the invariant sound: if no temporal path exists even without visited-set
restrictions, then no simple temporal cycle can be completed from that state.

The current implementation resets closing-time state for each start edge event.
That keeps window-specific pruning exact and leaves room for later path
bundling or shared preprocessing when profiling shows it is worthwhile.

## Path Bundling

The temporal Johnson search now walks vertex paths while carrying a bundle of
arrival timestamps for the current vertex. Each bundle entry stores:

```text
(arrival_timestamp, multiplicity)
```

When the search extends a vertex path through an edge, all compatible edge
timestamps are grouped by timestamp value and their multiplicities are summed.
Future reachability depends only on the latest timestamp, so assignments that
arrive at the same vertex with the same timestamp can be counted together.

This preserves duplicate timestamp multiplicity. If an input edge contains two
events with the same timestamp, both events contribute to the multiplicity even
though the bundled state stores one timestamp key.

The optimization is intentionally local to a single DFS path. It reduces
recursive branching over timestamp assignments without changing the vertex-path
duplicate rules or the histogram contract.
