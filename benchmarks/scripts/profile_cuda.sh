#!/usr/bin/env bash
# Profile a CUDA cycle-enum run with Nsight Systems (and optionally Compute).
#
# The project marks transfer, kernel, and reduction phases with NVTX ranges when
# built with NVTX, so an Nsight Systems timeline separates them. This script runs
# one configuration under nsys, and optionally one kernel under ncu.
#
# Environment variables:
#   BUILD_DIR        CUDA build directory (default build-h100)
#   INPUT            Graph input path (default the reference sample)
#   BACKEND          cycle-enum backend (default cuda)
#   ALGORITHM        Algorithm (default johnson)
#   MODE             Mode (default simple)
#   MAX_CYCLE_LENGTH Maximum cycle length (default 8)
#   TIME_WINDOW      Time window for windowed/temporal modes (default 3600)
#   CUDA_DEVICE      Device ordinal (default 0)
#   REPORT_DIR       Output directory (default benchmarks/results/profiles)
#   RUN_NCU          Set to 1 to also run an Nsight Compute kernel profile

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
build_dir="${BUILD_DIR:-${repo_root}/build-h100}"
input="${INPUT:-${repo_root}/tests/data/reference_sample.txt}"
backend="${BACKEND:-cuda}"
algorithm="${ALGORITHM:-johnson}"
mode="${MODE:-simple}"
max_cycle_length="${MAX_CYCLE_LENGTH:-8}"
time_window="${TIME_WINDOW:-3600}"
cuda_device="${CUDA_DEVICE:-0}"
report_dir="${REPORT_DIR:-${repo_root}/benchmarks/results/profiles}"
run_ncu="${RUN_NCU:-0}"

cli="${build_dir}/cycle-enum"
if [[ ! -x "${cli}" ]]; then
  printf 'error: %s not found; build the CUDA target first\n' "${cli}" >&2
  exit 1
fi

mkdir -p "${report_dir}"
stamp="$(date -u +%Y%m%dT%H%M%SZ)"

cli_args=(
  --input "${input}"
  --backend "${backend}"
  --algorithm "${algorithm}"
  --mode "${mode}"
  --max-cycle-length "${max_cycle_length}"
  --time-window "${time_window}"
  --cuda-device "${cuda_device}"
)

nsys profile \
  --trace=cuda,nvtx \
  --force-overwrite=true \
  --output "${report_dir}/nsys-${mode}-${stamp}" \
  "${cli}" "${cli_args[@]}"

printf 'Nsight Systems report written to %s\n' \
  "${report_dir}/nsys-${mode}-${stamp}.nsys-rep"

if [[ "${run_ncu}" == "1" ]]; then
  ncu \
    --set basic \
    --force-overwrite \
    --export "${report_dir}/ncu-${mode}-${stamp}" \
    "${cli}" "${cli_args[@]}"
  printf 'Nsight Compute report written to %s\n' \
    "${report_dir}/ncu-${mode}-${stamp}.ncu-rep"
fi
