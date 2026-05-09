# Temporal Cycle-Union Preprocessing

Cycle-union preprocessing builds a candidate vertex set for one temporal start
edge event. The set is used later as a pruning filter: vertices outside the set
cannot participate in a temporal cycle for that start event under the current
conservative test.

For a request `(root, first_vertex, start_timestamp, window_width)`, the current
implementation computes:

```text
forward = vertices reachable from first_vertex inside the start window
reverse = vertices that can reach root inside the start window
cycle_union = forward intersect reverse
```

Both reachability passes require at least one edge event in
`(start_timestamp, start_timestamp + window_width]`, but they ignore timestamp
ordering between later edges. Ignoring later ordering makes the set a superset
of valid temporal-cycle vertices, which is the safe direction for pruning.

This first version is deliberately sequential and byte-set based. It is meant
to establish correctness and test fixtures before the same idea is reused as a
host-side CUDA prefilter or moved into a GPU kernel if profiling justifies it.
