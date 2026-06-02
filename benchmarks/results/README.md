# Benchmark Results

This directory holds generated benchmark CSVs and plots. Its contents are
git-ignored except this README, so result files never enter version control;
keep published results in the experiment log or paper appendix instead.

## What lands here

- `*.csv` — raw rows from `run_cycle_benchmarks.py` and the matrix runner, plus
  summaries from `collect_results.py` and speedup tables from `plot_results.py`.
- `plots/` — runtime bar charts when matplotlib is available.
- `profiles/` — Nsight Systems and Nsight Compute reports.

## Recording a result set

For each result set, capture alongside the CSV (in the experiment log):

- graph name and size: vertices, directed edges, timestamped events;
- build configurations of this project and, if compared, the TBB baseline;
- CPU model, GPU model, compiler version, and CUDA version;
- thread counts and the CUDA launch configuration;
- the Slurm allocation details.

## Correctness before performance

Every summary carries the total cycle count. Before trusting any speedup, confirm
the total cycle count is identical across backends for the same configuration, and
run `scripts/compare_reference.sh` so the project and the TBB baseline histograms
match. A faster run with a different histogram is a defect, not a result.
