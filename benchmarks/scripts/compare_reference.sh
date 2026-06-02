#!/usr/bin/env bash
# Compare the project CLI and the SPAA/TBB baseline histograms on one input.
#
# Runs both programs on the same graph and mode, normalizes each cycle histogram
# to "length count" pairs, and reports MATCH or MISMATCH. This is the correctness
# gate that must pass before any performance comparison is trusted. It is kept
# outside the build so the project never depends on TBB or MPI.
#
# Environment variables:
#   CLI              Project cycle-enum path (default build/cycle-enum)
#   TBB_EXE          Baseline cycle executable (required)
#   INPUT            Graph input path (default the reference sample)
#   MODE             Project mode (default temporal)
#   ALGORITHM        Project algorithm (default johnson)
#   TIME_WINDOW      Time window seconds (default 3600)
#   MAX_CYCLE_LENGTH Project maximum cycle length (default 8)
#   TBB_ALGO         Baseline algorithm id (default 5, fine temporal Johnson)

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cli="${CLI:-${repo_root}/build/cycle-enum}"
tbb_exe="${TBB_EXE:-}"
input="${INPUT:-${repo_root}/tests/data/reference_sample.txt}"
mode="${MODE:-temporal}"
algorithm="${ALGORITHM:-johnson}"
time_window="${TIME_WINDOW:-3600}"
max_cycle_length="${MAX_CYCLE_LENGTH:-8}"
tbb_algo="${TBB_ALGO:-5}"

if [[ -z "${tbb_exe}" ]]; then
  printf 'error: set TBB_EXE to the baseline cycle executable\n' >&2
  exit 2
fi
for binary in "${cli}" "${tbb_exe}"; do
  if [[ ! -x "${binary}" ]]; then
    printf 'error: not executable: %s\n' "${binary}" >&2
    exit 2
  fi
done
if [[ ! -f "${input}" ]]; then
  printf 'error: input not found: %s\n' "${input}" >&2
  exit 2
fi

# Normalize any "length, count" or "length count" histogram lines (excluding the
# Total line and comments) to a sorted "length count" stream for comparison.
normalize() {
  awk '
    /^[[:space:]]*#/ { next }
    tolower($0) ~ /total/ { next }
    {
      gsub(/,/, " ")
      if (NF >= 2 && $1 ~ /^[0-9]+$/ && $2 ~ /^[0-9]+$/) {
        print $1, $2
      }
    }
  ' | sort -n
}

project_out="$("${cli}" \
  --input "${input}" \
  --algorithm "${algorithm}" \
  --mode "${mode}" \
  --time-window "${time_window}" \
  --max-cycle-length "${max_cycle_length}")"

tbb_out="$("${tbb_exe}" -i "${input}" -a "${tbb_algo}" -tws "${time_window}")"

project_norm="$(printf '%s\n' "${project_out}" | normalize)"
tbb_norm="$(printf '%s\n' "${tbb_out}" | normalize)"

if [[ "${project_norm}" == "${tbb_norm}" ]]; then
  printf 'MATCH: project and baseline histograms agree for mode=%s\n' "${mode}"
  exit 0
fi

printf 'MISMATCH for mode=%s\n--- project ---\n%s\n--- baseline ---\n%s\n' \
  "${mode}" "${project_norm}" "${tbb_norm}" >&2
exit 1
