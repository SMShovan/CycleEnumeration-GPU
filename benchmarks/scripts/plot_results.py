#!/usr/bin/env python3
"""Analyze cycle-enum benchmark CSVs: speedup table and optional plots.

Reads one or more CSV files from run_cycle_benchmarks.py, computes the median
elapsed time per configuration, and derives speedup relative to the
single-thread sequential baseline of the same input, algorithm, and mode. The
speedup table is always written as CSV. Runtime and speedup bar charts are also
written when matplotlib is available; without it the script still produces the
table and prints a note, so it is useful on a bare cluster allocation.
"""

import argparse
import csv
import os
import statistics
import sys
from collections import defaultdict

CONFIG_FIELDS = [
    "input",
    "algorithm",
    "mode",
    "time_window_seconds",
    "max_cycle_length",
]


def backend_label(row):
    backend = row.get("backend", "")
    threads = row.get("threads", "")
    if backend == "openmp" and threads:
        return f"openmp-{threads}"
    if backend == "tbb":
        algo = row.get("tbb_algo", "")
        return f"tbb-{algo}" if algo else "tbb"
    return backend


def load_medians(paths, include_failures):
    samples = defaultdict(list)
    totals = {}
    for path in paths:
        with open(path, newline="") as handle:
            for row in csv.DictReader(handle):
                if not include_failures and row.get("status") != "ok":
                    continue
                try:
                    elapsed = float(row["elapsed_ms"])
                except (KeyError, ValueError):
                    continue
                config = tuple(row.get(field, "") for field in CONFIG_FIELDS)
                key = (config, backend_label(row))
                samples[key].append(elapsed)
                totals[key] = row.get("total_cycles", "")
    medians = {key: statistics.median(values) for key, values in samples.items()}
    return medians, totals


def build_table(medians, totals):
    rows = []
    for (config, label), median_ms in sorted(medians.items()):
        baseline = medians.get((config, "sequential"))
        speedup = baseline / median_ms if baseline and median_ms else ""
        rows.append(
            list(config)
            + [
                label,
                f"{median_ms:.3f}",
                f"{speedup:.3f}" if speedup != "" else "",
                totals[(config, label)],
            ]
        )
    return rows


def maybe_plot(medians, output_dir):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; wrote the speedup table only")
        return

    configs = defaultdict(dict)
    for (config, label), median_ms in medians.items():
        configs[config][label] = median_ms

    os.makedirs(output_dir, exist_ok=True)
    for config, by_label in configs.items():
        labels = sorted(by_label)
        values = [by_label[label] for label in labels]
        fig, axis = plt.subplots()
        axis.bar(labels, values)
        axis.set_ylabel("median elapsed (ms)")
        axis.set_title(" ".join(part for part in config if part))
        axis.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        name = "_".join(part for part in config if part).replace("/", "_")
        fig.savefig(os.path.join(output_dir, f"runtime_{name}.png"))
        plt.close(fig)
    print(f"Wrote plots to {output_dir}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="+", help="benchmark CSV files")
    parser.add_argument("--output", help="speedup table CSV path (default stdout)")
    parser.add_argument("--plot-dir", help="directory for PNG plots")
    parser.add_argument("--include-failures", action="store_true")
    args = parser.parse_args()

    medians, totals = load_medians(args.csv, args.include_failures)
    table = build_table(medians, totals)

    out_handle = open(args.output, "w", newline="") if args.output else sys.stdout
    try:
        writer = csv.writer(out_handle)
        writer.writerow(
            CONFIG_FIELDS
            + ["backend", "median_ms", "speedup_vs_sequential", "total_cycles"]
        )
        writer.writerows(table)
    finally:
        if out_handle is not sys.stdout:
            out_handle.close()

    if args.plot_dir:
        maybe_plot(medians, args.plot_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
