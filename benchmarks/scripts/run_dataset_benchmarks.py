#!/usr/bin/env python3
"""Run the CUDA static cycle counter over every dataset and record results.

For each graph under the dataset directory this script invokes the optimized
work-queue CUDA backend (`--mode simple`, Johnson) with kernel timing enabled,
then records:

  * one timing row per dataset in   results/timings/<timestamp>.csv
  * the full histogram per dataset in results/histograms/<timestamp>.csv

Both files in a single run share the same date-time name so a timing file and
its histogram file can be paired. Datasets are processed smallest to largest
(by file size) and each result is flushed as soon as the dataset finishes, so
progress can be watched live and a long-running graph never hides the smaller
results that already completed. A second invocation writes new timestamped
files and never overwrites earlier data.

Raw network-repository edge lists are accepted directly: the CLI parser handles
comma- or whitespace-separated rows and fills a constant timestamp when the
column is absent (valid because static counting ignores timestamps).
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import re
import subprocess
import sys
from pathlib import Path

# Patterns over the CLI's stderr timing report (one "key: value" per line).
_STDERR_PATTERNS = {
    "vertices": re.compile(r"^vertices:\s*(\d+)", re.MULTILINE),
    "edges": re.compile(r"^edges:\s*(\d+)", re.MULTILINE),
    "kernel_ms": re.compile(r"^kernel_ms:\s*([0-9.eE+-]+)", re.MULTILINE),
    "memcpy_ms": re.compile(r"^memcpy_ms:\s*([0-9.eE+-]+)", re.MULTILINE),
    "total_ms": re.compile(r"^total_ms:\s*([0-9.eE+-]+)", re.MULTILINE),
}

TIMING_FIELDS = [
    "dataset",
    "vertices",
    "edges",
    "max_cycle_length",
    "kernel_ms",
    "memcpy_ms",
    "total_ms",
    "total_cycles",
    "status",
]
HISTOGRAM_FIELDS = ["dataset", "cycle_size", "num_of_cycles"]


def discover_datasets(dataset_dir: Path) -> list[Path]:
    """Return dataset edge-list files sorted by ascending file size."""
    files = sorted(dataset_dir.rglob("*.edges"), key=lambda p: p.stat().st_size)
    return files


def parse_histogram(stdout: str) -> list[tuple[str, str]]:
    """Parse the CLI histogram CSV into (cycle_size, num_of_cycles) rows."""
    rows: list[tuple[str, str]] = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 2:
            rows.append((parts[0], parts[1]))
    return rows


def run_dataset(cli: str, dataset: Path, args) -> dict:
    """Run the CUDA counter on one dataset and return parsed results."""
    cmd = [
        cli,
        "--input", str(dataset),
        "--backend", "cuda",
        "--mode", "simple",
        "--algorithm", "johnson",
        "--cuda-scheduler", "work-queue",
        "--cuda-device", str(args.cuda_device),
        "--max-cycle-length", str(args.max_cycle_length),
        "--report-timing",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)

    result = {field: "" for field in TIMING_FIELDS}
    result["dataset"] = dataset.stem
    result["max_cycle_length"] = str(args.max_cycle_length)
    result["histogram"] = []

    if proc.returncode != 0:
        result["status"] = "error"
        result["error"] = proc.stderr.strip().splitlines()[-1] if proc.stderr else "failed"
        return result

    for key, pattern in _STDERR_PATTERNS.items():
        match = pattern.search(proc.stderr)
        if match:
            result[key] = match.group(1)

    histogram = parse_histogram(proc.stdout)
    result["histogram"] = histogram
    total = next((count for length, count in histogram if length == "Total"), "")
    result["total_cycles"] = total
    result["status"] = "ok"
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cli", default="build/cycle-enum",
                        help="path to the cycle-enum binary (CUDA build)")
    parser.add_argument("--dataset-dir", default="dataset",
                        help="directory searched recursively for *.edges files")
    parser.add_argument("--results-dir", default="results",
                        help="root directory for timing and histogram CSVs")
    parser.add_argument("--max-cycle-length", type=int, default=6)
    parser.add_argument("--cuda-device", type=int, default=0)
    args = parser.parse_args()

    cli = args.cli
    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.is_dir():
        print(f"dataset directory not found: {dataset_dir}", file=sys.stderr)
        return 1

    datasets = discover_datasets(dataset_dir)
    if not datasets:
        print(f"no *.edges datasets under {dataset_dir}", file=sys.stderr)
        return 1

    stamp = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    timings_dir = Path(args.results_dir) / "timings"
    histograms_dir = Path(args.results_dir) / "histograms"
    timings_dir.mkdir(parents=True, exist_ok=True)
    histograms_dir.mkdir(parents=True, exist_ok=True)
    timings_path = timings_dir / f"{stamp}.csv"
    histograms_path = histograms_dir / f"{stamp}.csv"

    print(f"datasets (small -> large): {[d.stem for d in datasets]}")
    print(f"timings   -> {timings_path}")
    print(f"histograms-> {histograms_path}\n")

    # Open both files for the whole run and flush after each dataset so progress
    # is visible live and partial results survive an interruption.
    with timings_path.open("w", newline="") as tf, \
            histograms_path.open("w", newline="") as hf:
        timing_writer = csv.DictWriter(tf, fieldnames=TIMING_FIELDS)
        timing_writer.writeheader()
        tf.flush()
        hist_writer = csv.writer(hf)
        hist_writer.writerow(HISTOGRAM_FIELDS)
        hf.flush()

        for index, dataset in enumerate(datasets, start=1):
            size_mb = dataset.stat().st_size / (1024 * 1024)
            print(f"[{index}/{len(datasets)}] {dataset.stem} "
                  f"({size_mb:.1f} MB) ... ", end="", flush=True)

            result = run_dataset(cli, dataset, args)

            timing_writer.writerow({f: result.get(f, "") for f in TIMING_FIELDS})
            tf.flush()
            for length, count in result.get("histogram", []):
                hist_writer.writerow([result["dataset"], length, count])
            hf.flush()

            if result["status"] == "ok":
                print(f"kernel={result['kernel_ms']}ms "
                      f"memcpy={result['memcpy_ms']}ms "
                      f"total={result['total_ms']}ms "
                      f"cycles={result['total_cycles']}")
            else:
                print(f"FAILED: {result.get('error', 'unknown error')}")

    print(f"\ndone. results in {timings_path} and {histograms_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
