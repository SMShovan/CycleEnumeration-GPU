#!/usr/bin/env bash
# Configure, build, and run a sequential-versus-OpenMP CPU scaling sweep.
#
# This wrapper drives benchmarks/run_cycle_benchmarks.py with a thread-count
# sweep so the OpenMP Johnson and Read-Tarjan baselines can be compared against
# the single-thread sequential path. It is CPU-only and does not require CUDA,
# so it runs on a developer workstation as well as a CPU cluster node.
#
# Configuration is environment-variable driven to match scripts/h100_smoke.sh:
#   BUILD_DIR          CMake build directory (default build-openmp)
#   JOBS               Parallel build jobs
#   THREAD_LIST        Space-separated OpenMP thread counts (default "1 2 4 8")
#   ALGORITHMS         Space-separated algorithms (default "johnson read-tarjan")
#   MODES              Space-separated modes (default the three CPU modes)
#   SAMPLE_INPUT       Graph input path (default the reference sample)
#   TIME_WINDOW        Time window in seconds for windowed and temporal modes
#   MAX_CYCLE_LENGTH   Maximum cycle length passed to the CLI
#   WARMUP / REPEAT    Warmup and measured repetitions per case
#   TIMEOUT            Per-run timeout in seconds
#   BENCHMARK_OUTPUT   CSV output path
#   OPENMP_CMAKE_ARGS  Extra CMake args (e.g. macOS libomp hints)

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
build_dir="${BUILD_DIR:-${repo_root}/build-openmp}"
parallel_jobs="${JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 8)}"
thread_list="${THREAD_LIST:-1 2 4 8}"
algorithms="${ALGORITHMS:-johnson read-tarjan}"
modes="${MODES:-simple simple-time-window temporal}"
sample_input="${SAMPLE_INPUT:-${repo_root}/tests/data/reference_sample.txt}"
time_window="${TIME_WINDOW:-3600}"
max_cycle_length="${MAX_CYCLE_LENGTH:-6}"
warmup="${WARMUP:-1}"
repeat="${REPEAT:-3}"
timeout_seconds="${TIMEOUT:-600}"
benchmark_output="${BENCHMARK_OUTPUT:-${repo_root}/benchmarks/results/openmp-scaling.csv}"

# shellcheck disable=SC2206
extra_cmake_args=(${OPENMP_CMAKE_ARGS:-})

cmake -S "${repo_root}" -B "${build_dir}" \
  -DCYCLE_ENUM_BUILD_TESTS=ON \
  -DCYCLE_ENUM_ENABLE_OPENMP=ON \
  -DCYCLE_ENUM_ENABLE_CUDA=OFF \
  -DCYCLE_ENUM_ENABLE_DOXYGEN=OFF \
  "${extra_cmake_args[@]}"

cmake --build "${build_dir}" --parallel "${parallel_jobs}"

cmd=(python3 "${repo_root}/benchmarks/run_cycle_benchmarks.py"
  --cycle-enum "${build_dir}/cycle-enum"
  --input "${sample_input}"
  --backend sequential
  --backend openmp
  --time-window "${time_window}"
  --max-cycle-length "${max_cycle_length}"
  --warmup "${warmup}"
  --repeat "${repeat}"
  --timeout "${timeout_seconds}"
  --output "${benchmark_output}")

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

printf 'OpenMP scaling benchmark written to %s\n' "${benchmark_output}"
