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

## OpenMP CPU Scaling Sweep

`scripts/run_openmp_benchmarks.sh` configures a CPU-only OpenMP build and runs
the sequential path against the OpenMP backend across a thread-count sweep,
writing one CSV. It is environment-variable driven, like the H100 script:

```sh
THREAD_LIST="1 2 4 8 16" \
ALGORITHMS="johnson read-tarjan" \
benchmarks/scripts/run_openmp_benchmarks.sh
```

On macOS, where Apple Clang needs explicit libomp hints, pass them through
`OPENMP_CMAKE_ARGS`:

```sh
OPENMP_CMAKE_ARGS="-DCMAKE_CXX_FLAGS=-I$(brew --prefix libomp)/include \
  -DOpenMP_CXX_FLAGS=-Xclang;-fopenmp -DOpenMP_CXX_LIB_NAMES=omp \
  -DOpenMP_omp_LIBRARY=$(brew --prefix libomp)/lib/libomp.dylib" \
benchmarks/scripts/run_openmp_benchmarks.sh
```

`BUILD_DIR`, `MODES`, `SAMPLE_INPUT`, `TIME_WINDOW`, `MAX_CYCLE_LENGTH`,
`WARMUP`, `REPEAT`, `TIMEOUT`, and `BENCHMARK_OUTPUT` override the rest of the
sweep.

## Cluster Sweep Example

For an allocated H100 node, `scripts/h100_smoke.sh` configures a CUDA/OpenMP
build, runs CTest, and writes a small CPU/GPU benchmark CSV. Environment
variables such as `BUILD_DIR`, `JOBS`, `OPENMP_THREADS`, `CUDA_DEVICE`,
`MAX_CYCLE_LENGTH`, `TIME_WINDOW`, `SAMPLE_INPUT`, and `BENCHMARK_OUTPUT` can
override the defaults.

The CUDA backend currently benchmarks bounded Johnson implementations for
static simple cycles, simple time-window cycles, and temporal cycles, so
`--max-cycle-length` is required. On an H100 build configured with
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
  --mode simple-time-window \
  --mode temporal \
  --time-window 3600 \
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

## Matrix Runner and Analysis

`scripts/run_benchmarks.sh` runs a configurable matrix (datasets x backends x
algorithms x modes x thread counts) into one CSV, failing fast if a dataset path
is missing:

```sh
DATASETS="benchmarks/data/graph-a.txt benchmarks/data/graph-b.txt" \
BACKENDS="sequential openmp cuda" \
MODES="simple temporal" \
OUTPUT=benchmarks/results/matrix.csv \
benchmarks/scripts/run_benchmarks.sh
```

`scripts/collect_results.py` aggregates one or more result CSVs into a
per-configuration summary with run count and min/median/mean elapsed time:

```sh
python3 benchmarks/scripts/collect_results.py benchmarks/results/matrix.csv
```

`scripts/plot_results.py` computes speedup relative to the single-thread
sequential baseline of the same input, algorithm, and mode, and writes runtime
bar charts when matplotlib is available:

```sh
python3 benchmarks/scripts/plot_results.py benchmarks/results/matrix.csv \
  --output benchmarks/results/speedup.csv \
  --plot-dir benchmarks/results/plots
```

The total cycle count travels through every summary so a configuration that
changes the answer is caught next to its timing.

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
