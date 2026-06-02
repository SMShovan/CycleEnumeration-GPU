#!/usr/bin/env python3
"""Aggregate cycle-enum benchmark CSVs into a per-configuration summary.

Reads one or more CSV files produced by run_cycle_benchmarks.py and groups the
measured rows by configuration (input, backend, algorithm, mode, threads, device,
window, max length). For each group it reports the run count and the minimum,
median, and mean elapsed time in milliseconds, plus the total cycle count, which
must be identical within a group. The summary is written as CSV to stdout or to
a file.

Dependency-free so it runs on a cluster allocation without extra packages.
"""

import argparse
import csv
import statistics
import sys
from collections import defaultdict

GROUP_KEYS = [
    "input",
    "backend",
    "algorithm",
    "mode",
    "threads",
    "cuda_device",
    "time_window_seconds",
    "max_cycle_length",
]


def load_rows(paths):
    for path in paths:
        with open(path, newline="") as handle:
            for row in csv.DictReader(handle):
                yield row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="+", help="benchmark CSV files to aggregate")
    parser.add_argument("--output", help="summary CSV path (default stdout)")
    parser.add_argument(
        "--include-failures",
        action="store_true",
        help="include rows whose status is not 'ok'",
    )
    args = parser.parse_args()

    groups = defaultdict(list)
    totals = {}
    for row in load_rows(args.csv):
        if not args.include_failures and row.get("status") != "ok":
            continue
        try:
            elapsed = float(row["elapsed_ms"])
        except (KeyError, ValueError):
            continue
        key = tuple(row.get(field, "") for field in GROUP_KEYS)
        groups[key].append(elapsed)
        totals[key] = row.get("total_cycles", "")

    out_handle = open(args.output, "w", newline="") if args.output else sys.stdout
    try:
        writer = csv.writer(out_handle)
        writer.writerow(
            GROUP_KEYS
            + ["runs", "min_ms", "median_ms", "mean_ms", "total_cycles"]
        )
        for key in sorted(groups):
            samples = groups[key]
            writer.writerow(
                list(key)
                + [
                    len(samples),
                    f"{min(samples):.3f}",
                    f"{statistics.median(samples):.3f}",
                    f"{statistics.fmean(samples):.3f}",
                    totals[key],
                ]
            )
    finally:
        if out_handle is not sys.stdout:
            out_handle.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
