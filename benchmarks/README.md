# Benchmark Harness

`run_cycle_benchmarks.py` records repeatable wall-clock measurements for the
project CLI and, when available, the SPAA 2022 TBB baseline executable. The
script is dependency-free so it can run on the local macOS development machine
and on a Linux cluster allocation without a Python environment setup step.

Each measured run appends one CSV row with the selected backend, algorithm,
mode, thread or device setting, elapsed wall time, parsed histogram, total
cycle count, and the exact command line. Failures are recorded as rows too,
which keeps long sweeps auditable when one backend is unavailable or one graph
configuration exceeds a timeout.

## Local Smoke Run

From the repository root:

```sh
python3 benchmarks/run_cycle_benchmarks.py \
  --cycle-enum build/cycle-enum \
  --input tests/data/reference_sample.txt \
  --backend sequential \
  --algorithm johnson \
  --mode simple-time-window \
  --time-window 3600 \
  --max-cycle-length 6 \
  --warmup 0 \
  --repeat 1 \
  --output benchmarks/results/local-smoke.csv
```

The same smoke case is also available through CMake when benchmark targets are
enabled:

```sh
cmake -S . -B build-bench -DCYCLE_ENUM_BUILD_BENCHMARKS=ON
cmake --build build-bench --target cycle_enum_benchmark_smoke
```

## Cluster Sweep Example

The CUDA backend currently benchmarks the bounded static Johnson implementation,
so `--max-cycle-length` is required. On an H100 build configured with
`-DCYCLE_ENUM_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=90`, a first comparison
sweep can be:

```sh
python3 benchmarks/run_cycle_benchmarks.py \
  --cycle-enum build-cuda/cycle-enum \
  --input /path/to/graph.txt \
  --backend sequential \
  --backend openmp \
  --backend cuda \
  --algorithm johnson \
  --mode simple \
  --openmp-threads 1 \
  --openmp-threads 16 \
  --openmp-threads 32 \
  --cuda-device 0 \
  --max-cycle-length 8 \
  --warmup 1 \
  --repeat 5 \
  --timeout 3600 \
  --output benchmarks/results/h100-static-johnson.csv
```

## TBB Baseline Example

After building the baseline repository, pass its `cycle` executable with one or
more algorithm ids. The harness uses `-tws` so the time window is always
interpreted as seconds.

```sh
python3 benchmarks/run_cycle_benchmarks.py \
  --cycle-enum build-cuda/cycle-enum \
  --tbb-exe ../parallel-cycle-enumeration/build/cycle \
  --input /path/to/graph.txt \
  --backend openmp \
  --algorithm johnson \
  --mode temporal \
  --time-window 3600 \
  --openmp-threads 32 \
  --tbb-algo 5 \
  --tbb-threads 32 \
  --warmup 1 \
  --repeat 5 \
  --timeout 7200 \
  --output benchmarks/results/tbb-temporal-johnson.csv
```

Baseline algorithm ids follow the original executable:

| id | Meaning |
| --- | --- |
| 0 | Coarse-grained Johnson with time window |
| 1 | Fine-grained Johnson with time window |
| 2 | Coarse-grained Read-Tarjan with time window |
| 3 | Fine-grained Read-Tarjan with time window |
| 4 | Coarse-grained temporal Johnson |
| 5 | Fine-grained temporal Johnson |
| 6 | Coarse-grained temporal Read-Tarjan |
| 7 | Fine-grained temporal Read-Tarjan |
| 8 | Single-thread temporal Read-Tarjan |
| 9 | 2SCENT without source detection |
| 10 | 2SCENT with source detection |
| 11 | Cycle-union preprocessing timing |

## Reporting Notes

For report-quality runs, keep the generated CSV with the graph name, build
configuration, GPU model, CPU model, compiler versions, CUDA version, and Slurm
allocation details. The CSV command column preserves the exact executable
arguments, but the surrounding machine metadata should be recorded in the
experiment log or paper appendix.
