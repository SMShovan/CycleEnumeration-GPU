# Benchmark Data Directory

This directory holds local benchmark graphs. Its contents are git-ignored except
for this README, so large datasets never enter version control.

Place temporal edge-list files here (`source target timestamp` per line, see
`../datasets.md`) and point the benchmark scripts at them, for example:

```sh
SAMPLE_INPUT=benchmarks/data/my-graph.txt \
benchmarks/scripts/run_benchmarks.sh
```

Record the provenance, size, and license of every dataset in the experiment log
or paper appendix; this directory deliberately keeps no such metadata so nothing
large or licensed is committed by accident.
