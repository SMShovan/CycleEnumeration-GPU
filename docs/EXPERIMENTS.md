# Recompute vs. Update Experiments

This guide describes how to compare full recomputation against the incremental
update of the static simple-cycle histogram. The incremental update applies a
batch of edge insertions and deletions to a known histogram without recomputing
the unchanged cycles; see `docs/design/dynamic_update.md` for the algorithm.

## What is measured

For a graph `G0` with known histogram `H0` and a batch `B`:

- **Recompute**: the time to recount the whole post-batch graph `G_final`.
- **Update**: the time to adjust `H0` to the histogram of `G_final` by touching
  only the cycles through the changed edges.

`H0` is treated as prior knowledge and is computed once, untimed. Both paths
produce the same histogram; the CLI's `--compare-recompute` verifies this on
every run (`match=yes`). A `match=no` is a correctness defect and invalidates the
timing.

## Single run

```sh
build/cycle-enum \
  --input path/to/graph.txt \
  --task update --backend sequential \
  --mode simple --max-cycle-length 8 \
  --deletes 8 --inserts 8 --batch-seed 1 \
  --compare-recompute
```

The histogram is printed to stdout; `deletions=`, `insertions=`,
`update_seconds=`, `recompute_seconds=`, and `match=` are printed to stderr.

Backends: `--backend sequential` (single thread), `--backend openmp
--openmp-threads N` (CPU scaling), `--backend cuda --cuda-device D` (single GPU).
All three apply delete-then-insert with edge-id ownership and produce identical
results; only the per-edge work is parallelized differently.

## Sweep

`benchmarks/scripts/run_dynamic_benchmarks.py` sweeps batch sizes, locality
windows, and seeds, recording update and recompute timings, their speedup, and
the correctness match in one CSV:

```sh
python3 benchmarks/scripts/run_dynamic_benchmarks.py \
  --cycle-enum build-cuda/cycle-enum \
  --input path/to/graph.txt \
  --backend cuda --cuda-device 0 \
  --max-cycle-length 8 \
  --deletes 1 2 4 8 16 \
  --inserts 1 2 4 8 16 \
  --locality 16 64 256 \
  --seeds 10 --repeat 5 \
  --output benchmarks/results/dynamic-cuda.csv
```

## Interpreting the results

The update wins when the batch touches only a small fraction of the graph's
cycles; it ties or loses when the changes hit cycle-dense regions or when the
graph is so small that fixed overhead dominates. The decisive variable is the
fraction of cycles a batch touches, which the **locality window** controls: a
small window concentrates changes (higher touch fraction) and a large window or
no window spreads them out. Sweep `--deletes`/`--inserts` (batch size) and
`--locality` (locality) and plot speedup against them; expect the speedup to
cross 1.0 as the touched fraction grows, with realistic streaming workloads
sitting well below the crossover.

## Notes for scalability runs

- Use a large, cycle-rich graph: the update's advantage grows with how expensive
  a full recompute is.
- Keep `--max-cycle-length` fixed across recompute and update; both honor the
  same cap and the prior histogram must use it too.
- For CPU scaling, sweep `--openmp-threads` for both the OpenMP update and the
  OpenMP recompute baseline.
- Record graph size, build configuration, CPU/GPU models, and the exact command
  with each result CSV.
