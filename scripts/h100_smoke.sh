#!/usr/bin/env bash
# Configure, build, test, and benchmark a small CUDA smoke run on an H100 node.
#
# This script is intended for an interactive Slurm allocation or a submitted
# batch job where CUDA, CMake, a C++ compiler, and Python 3 are already loaded.
# It deliberately keeps paths relative to the repository root so copied cluster
# workspaces remain reproducible.

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${BUILD_DIR:-${repo_root}/build-h100}"
parallel_jobs="${JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 8)}"
benchmark_output="${BENCHMARK_OUTPUT:-${repo_root}/benchmarks/results/h100-smoke.csv}"
sample_input="${SAMPLE_INPUT:-${repo_root}/tests/data/reference_sample.txt}"
max_cycle_length="${MAX_CYCLE_LENGTH:-6}"
time_window="${TIME_WINDOW:-3600}"
openmp_threads="${OPENMP_THREADS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 8)}"
cuda_device="${CUDA_DEVICE:-0}"

cmake -S "${repo_root}" -B "${build_dir}" \
  -DCYCLE_ENUM_BUILD_TESTS=ON \
  -DCYCLE_ENUM_ENABLE_OPENMP=ON \
  -DCYCLE_ENUM_ENABLE_CUDA=ON \
  -DCYCLE_ENUM_ENABLE_DOXYGEN=OFF \
  -DCMAKE_CUDA_ARCHITECTURES=90

cmake --build "${build_dir}" --parallel "${parallel_jobs}"
ctest --test-dir "${build_dir}" --output-on-failure

python3 "${repo_root}/benchmarks/run_cycle_benchmarks.py" \
  --cycle-enum "${build_dir}/cycle-enum" \
  --input "${sample_input}" \
  --backend sequential \
  --backend openmp \
  --backend cuda \
  --algorithm johnson \
  --mode simple \
  --mode simple-time-window \
  --mode temporal \
  --time-window "${time_window}" \
  --max-cycle-length "${max_cycle_length}" \
  --openmp-threads 1 \
  --openmp-threads "${openmp_threads}" \
  --cuda-device "${cuda_device}" \
  --warmup 1 \
  --repeat 3 \
  --timeout 600 \
  --output "${benchmark_output}"

printf 'H100 smoke benchmark written to %s\n' "${benchmark_output}"
