#!/usr/bin/env bash
# Run a configurable benchmark matrix across datasets, backends, and modes.
#
# This is the general driver around benchmarks/run_cycle_benchmarks.py. It checks
# that every dataset exists before running, so a missing graph stops the sweep
# instead of producing partial results, then runs the full matrix into one CSV.
#
# Environment variables (space-separated lists where noted):
#   BUILD_DIR          Build directory holding cycle-enum (default build)
#   DATASETS           Dataset paths (default the reference sample)
#   BACKENDS           Backends to run (default "sequential openmp")
#   ALGORITHMS         Algorithms (default "johnson read-tarjan")
#   MODES              Modes (default "simple temporal")
#   THREAD_LIST        OpenMP thread counts (default "1 4 16")
#   CUDA_DEVICE        CUDA device ordinal (default 0)
#   MAX_CYCLE_LENGTH   Maximum cycle length (default 8)
#   TIME_WINDOW        Time window seconds (default 3600)
#   WARMUP / REPEAT    Warmup and measured repetitions (default 1 / 5)
#   TIMEOUT            Per-run timeout seconds (default 3600)
#   OUTPUT             CSV output path

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
build_dir="${BUILD_DIR:-${repo_root}/build}"
datasets="${DATASETS:-${repo_root}/tests/data/reference_sample.txt}"
backends="${BACKENDS:-sequential openmp}"
algorithms="${ALGORITHMS:-johnson read-tarjan}"
modes="${MODES:-simple temporal}"
thread_list="${THREAD_LIST:-1 4 16}"
cuda_device="${CUDA_DEVICE:-0}"
max_cycle_length="${MAX_CYCLE_LENGTH:-8}"
time_window="${TIME_WINDOW:-3600}"
warmup="${WARMUP:-1}"
repeat="${REPEAT:-5}"
timeout_seconds="${TIMEOUT:-3600}"
output="${OUTPUT:-${repo_root}/benchmarks/results/matrix.csv}"

cli="${build_dir}/cycle-enum"
if [[ ! -x "${cli}" ]]; then
  printf 'error: %s not found; build the project first\n' "${cli}" >&2
  exit 1
fi

for dataset in ${datasets}; do
  if [[ ! -f "${dataset}" ]]; then
    printf 'error: dataset not found: %s\n' "${dataset}" >&2
    exit 1
  fi
done

cmd=(python3 "${repo_root}/benchmarks/run_cycle_benchmarks.py"
  --cycle-enum "${cli}"
  --time-window "${time_window}"
  --max-cycle-length "${max_cycle_length}"
  --cuda-device "${cuda_device}"
  --warmup "${warmup}"
  --repeat "${repeat}"
  --timeout "${timeout_seconds}"
  --allow-failures
  --output "${output}")

for dataset in ${datasets}; do
  cmd+=(--input "${dataset}")
done
for backend in ${backends}; do
  cmd+=(--backend "${backend}")
done
for algorithm in ${algorithms}; do
  cmd+=(--algorithm "${algorithm}")
done
for mode in ${modes}; do
  cmd+=(--mode "${mode}")
done
for threads in ${thread_list}; do
  cmd+=(--openmp-threads "${threads}")
done

"${cmd[@]}"

printf 'Benchmark matrix written to %s\n' "${output}"
printf 'Summarize it with: python3 %s %s\n' \
  "${repo_root}/benchmarks/scripts/collect_results.py" "${output}"
