#!/usr/bin/env python3
"""Sweep the rank (proposed) CUDA backend over every dataset and record timing
and cycle counts for later plotting.

For each graph under the dataset directory, and for each requested max cycle
length, this runs the rank backend with device timing enabled:

  cycle-enum --backend cuda --mode simple --algorithm johnson \\
             --cuda-scheduler work-queue --blocking rank \\
             --max-cycle-length L [--step-budget N] --report-timing

and records one timing row plus the per-length histogram.

  * timings   -> <results>/timings/<timestamp>.csv     (one row per run)
  * histograms-> <results>/histograms/<timestamp>.csv  (per cycle length)

With --step-budget 0 (default) the rank backend is exact, so status is "exact".
With a positive budget it may stop heavy roots and report a lower bound, so
status is "lower-bound" and flagged_roots is the number of stopped roots.

A per-dataset --timeout (seconds) stops a graph that runs too long so the sweep
keeps going; that run is recorded with status "timeout". Datasets are processed
smallest to largest by file size, and every row is flushed immediately, so
progress is visible live and partial results survive an interruption.
"""

import argparse
import csv
import datetime as dt
import re
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Tuple

# Timing report is printed to stderr (one "key: value" per line) by --report-timing.
_STDERR_PATTERNS = {
    "vertices": re.compile(r"^vertices:\s*(\d+)", re.MULTILINE),
    "edges": re.compile(r"^edges:\s*(\d+)", re.MULTILINE),
    "kernel_ms": re.compile(r"^kernel_ms:\s*([0-9.eE+-]+)", re.MULTILINE),
    "memcpy_ms": re.compile(r"^memcpy_ms:\s*([0-9.eE+-]+)", re.MULTILINE),
    "total_ms": re.compile(r"^total_ms:\s*([0-9.eE+-]+)", re.MULTILINE),
    "flagged_roots": re.compile(r"^flagged_roots:\s*(\d+)", re.MULTILINE),
}
# Footer comment on stdout: "# status: exact (...)" or "# status: lower-bound (...)".
_STATUS_PATTERN = re.compile(r"^#\s*status:\s*([\w-]+)", re.MULTILINE)

TIMING_FIELDS = [
    "dataset", "vertices", "edges", "max_cycle_length", "step_budget",
    "kernel_ms", "memcpy_ms", "total_ms", "total_cycles", "status",
    "flagged_roots", "wall_seconds", "command",
]
HISTOGRAM_FIELDS = ["dataset", "max_cycle_length", "cycle_size", "num_of_cycles"]


def discover_datasets(dataset_dir: Path) -> List[Path]:
    """Return *.edges and *.mtx files sorted by ascending file size."""
    files = []  # type: List[Path]
    for pattern in ("*.edges", "*.mtx"):
        files.extend(dataset_dir.rglob(pattern))
    return sorted(set(files), key=lambda p: p.stat().st_size)


def parse_hist(stdout: str) -> List[Tuple[str, str]]:
    """Parse the CLI histogram CSV (ignores '#' comment and status lines)."""
    rows = []  # type: List[Tuple[str, str]]
    for line in stdout.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 2:
            rows.append((parts[0], parts[1]))
    return rows


def run_one(cli: str, dataset: Path, max_length: int, args) -> Dict:
    """Run the rank backend on one (dataset, length) and return parsed results."""
    cmd = [
        cli, "--input", str(dataset),
        "--backend", "cuda", "--mode", "simple", "--algorithm", "johnson",
        "--cuda-scheduler", "work-queue", "--blocking", "rank",
        "--max-cycle-length", str(max_length),
        "--cuda-device", str(args.cuda_device),
        "--report-timing",
    ]
    if args.step_budget > 0:
        cmd += ["--step-budget", str(args.step_budget)]

    result = {field: "" for field in TIMING_FIELDS}
    result["dataset"] = dataset.stem
    result["max_cycle_length"] = str(max_length)
    result["step_budget"] = str(args.step_budget)
    result["command"] = " ".join(cmd)
    result["histogram"] = []  # type: ignore[assignment]

    start = time.monotonic()
    try:
        proc = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            universal_newlines=True,
            timeout=args.timeout if args.timeout > 0 else None,
        )
    except subprocess.TimeoutExpired:
        result["status"] = "timeout"
        result["wall_seconds"] = f"{time.monotonic() - start:.3f}"
        return result

    result["wall_seconds"] = f"{time.monotonic() - start:.3f}"
    if proc.returncode != 0:
        result["status"] = "error"
        tail = proc.stderr.strip().splitlines()
        result["error"] = tail[-1] if tail else "failed"
        return result

    for key, pattern in _STDERR_PATTERNS.items():
        match = pattern.search(proc.stderr)
        if match:
            result[key] = match.group(1)

    histogram = parse_hist(proc.stdout)
    result["histogram"] = histogram  # type: ignore[assignment]
    result["total_cycles"] = next((c for s, c in histogram if s == "Total"), "")
    status = _STATUS_PATTERN.search(proc.stdout)
    result["status"] = status.group(1) if status else "ok"
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cli", default="build-cuda/cycle-enum",
                        help="path to the cycle-enum binary (CUDA build)")
    parser.add_argument("--dataset-dir", default="dataset",
                        help="directory searched recursively for *.edges/*.mtx")
    parser.add_argument("--results-dir", default="benchmarks/results/rank",
                        help="root directory for timing and histogram CSVs")
    parser.add_argument("--lengths", default="6",
                        help="comma-separated max cycle lengths, e.g. 4,5,6")
    parser.add_argument("--step-budget", type=int, default=0,
                        help="per-root work cap; 0 = off (exact)")
    parser.add_argument("--timeout", type=float, default=0,
                        help="per-run wall-clock cap in seconds; 0 = no limit")
    parser.add_argument("--cuda-device", type=int, default=0)
    args = parser.parse_args()

    lengths = [int(x) for x in args.lengths.split(",") if x.strip()]
    if not lengths:
        print("no --lengths given")
        return 1

    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.is_dir():
        print(f"dataset directory not found: {dataset_dir}")
        return 1
    datasets = discover_datasets(dataset_dir)
    if not datasets:
        print(f"no *.edges or *.mtx datasets under {dataset_dir}")
        return 1

    stamp = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_dir = Path(args.results_dir)
    (results_dir / "timings").mkdir(parents=True, exist_ok=True)
    (results_dir / "histograms").mkdir(parents=True, exist_ok=True)
    timings_path = results_dir / "timings" / f"{stamp}.csv"
    histograms_path = results_dir / "histograms" / f"{stamp}.csv"

    print(f"rank sweep: {len(datasets)} datasets x lengths {lengths} "
          f"(step_budget={args.step_budget}, timeout={args.timeout}s)")
    print(f"timings    -> {timings_path}")
    print(f"histograms -> {histograms_path}\n")

    with timings_path.open("w", newline="") as tf, \
            histograms_path.open("w", newline="") as hf:
        timing_writer = csv.DictWriter(tf, fieldnames=TIMING_FIELDS)
        timing_writer.writeheader()
        tf.flush()
        hist_writer = csv.writer(hf)
        hist_writer.writerow(HISTOGRAM_FIELDS)
        hf.flush()

        total = len(datasets) * len(lengths)
        index = 0
        for dataset in datasets:
            size_mb = dataset.stat().st_size / (1024 * 1024)
            for max_length in lengths:
                index += 1
                print(f"[{index}/{total}] {dataset.stem} ({size_mb:.1f} MB) "
                      f"L={max_length} ... ", end="", flush=True)
                result = run_one(args.cli, dataset, max_length, args)

                timing_writer.writerow({f: result.get(f, "") for f in TIMING_FIELDS})
                tf.flush()
                for cycle_size, count in result.get("histogram", []):
                    hist_writer.writerow([result["dataset"], max_length,
                                          cycle_size, count])
                hf.flush()

                if result["status"] in ("ok", "exact", "lower-bound"):
                    print(f"{result['status']} cycles={result['total_cycles']} "
                          f"total_ms={result['total_ms']} "
                          f"wall={result['wall_seconds']}s "
                          f"flagged={result.get('flagged_roots', '')}")
                else:
                    print(f"{result['status'].upper()} "
                          f"{result.get('error', '')} "
                          f"wall={result['wall_seconds']}s")

    print(f"\ndone.\n  timings    -> {timings_path}\n  histograms -> {histograms_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
