# Benchmark Datasets

This document describes the datasets used to benchmark CycleEnumeration-GPU and
the rules for managing them. Large graphs are kept out of Git; only the tiny
reference sample ships with the repository.

## Input format

Every dataset is a temporal edge list, one directed edge per line:

```text
source target timestamp
```

- Lines beginning with `#` or `%` are comments.
- Self-loops are ignored by the parser.
- Vertex ids are arbitrary non-negative integers and are compacted internally.
- Timestamps are integers; duplicate timestamps on an edge are preserved.

This is the same format the project CLI and the SPAA/TBB baseline read, so a
dataset can be benchmarked against both without conversion.

## In-repository fixtures

| File | Vertices | Purpose |
| --- | --- | --- |
| `tests/data/reference_sample.txt` | small | Smoke runs and the histogram example in the README |
| `tests/data/sample_temporal.txt` | small | Parser and temporal fixtures |

These are intentionally tiny. They validate that the harness and CLI run; they
are not performance datasets.

## External datasets

Performance datasets are not committed. Place them under `benchmarks/data/`,
which is git-ignored, or point the harness at an absolute path. For each external
dataset record, in the experiment log:

- the source and license of the graph,
- vertex count, directed edge count, and timestamped event count,
- the time span of the timestamps, and
- any preprocessing applied to reach the `source target timestamp` format.

Suggested public temporal graphs for cycle enumeration experiments include
financial transaction graphs and interaction networks with timestamps; the exact
set is recorded with the published results so runs are reproducible.

## Missing-dataset behavior

The benchmark scripts fail clearly when a dataset path does not exist rather than
silently skipping it, so a sweep that references a missing graph stops instead of
producing partial, misleading results.
