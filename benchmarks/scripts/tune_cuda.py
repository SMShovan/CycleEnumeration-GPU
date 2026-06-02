#!/usr/bin/env python3
"""Sweep CUDA work-queue launch parameters and record timings.

The persistent work-queue kernel reads its block size and resident blocks per
multiprocessor from the environment (CYCLE_ENUM_CUDA_BLOCK_SIZE and
CYCLE_ENUM_CUDA_BLOCKS_PER_SM). This script sweeps those values, runs the
cycle-enum CLI for each combination, and writes one CSV row per combination with
the best observed wall-clock time and the total cycle count, so the best default
can be chosen from data.

The script is dependency-free so it runs on a cluster allocation without a
Python environment setup step.
"""

import argparse
import csv
import os
import subprocess
import sys
import time


def parse_total_cycles(output: str) -> int:
    total = 0
    for line in output.splitlines():
        line = line.strip()
        if line.lower().startswith("total"):
            parts = line.split(",")
            if len(parts) == 2:
                try:
                    total = int(parts[1].strip())
                except ValueError:
                    pass
    return total


def run_once(cli, args, env):
    start = time.perf_counter()
    completed = subprocess.run(
        [cli, *args],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    elapsed = time.perf_counter() - start
    return elapsed, parse_total_cycles(completed.stdout)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cycle-enum", required=True, help="cycle-enum CLI path")
    parser.add_argument("--input", required=True, help="graph input path")
    parser.add_argument("--algorithm", default="johnson")
    parser.add_argument("--mode", default="simple")
    parser.add_argument("--max-cycle-length", type=int, default=8)
    parser.add_argument("--time-window", type=int, default=3600)
    parser.add_argument("--cuda-device", type=int, default=0)
    parser.add_argument(
        "--block-sizes", type=int, nargs="+", default=[64, 128, 256, 512]
    )
    parser.add_argument(
        "--blocks-per-sm", type=int, nargs="+", default=[8, 16, 32]
    )
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--output", required=True, help="CSV output path")
    args = parser.parse_args()

    cli_args = [
        "--input", args.input,
        "--backend", "cuda",
        "--algorithm", args.algorithm,
        "--mode", args.mode,
        "--max-cycle-length", str(args.max_cycle_length),
        "--time-window", str(args.time_window),
        "--cuda-device", str(args.cuda_device),
    ]

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["block_size", "blocks_per_sm", "best_seconds", "total_cycles"]
        )

        for block_size in args.block_sizes:
            for blocks_per_sm in args.blocks_per_sm:
                env = dict(os.environ)
                env["CYCLE_ENUM_CUDA_BLOCK_SIZE"] = str(block_size)
                env["CYCLE_ENUM_CUDA_BLOCKS_PER_SM"] = str(blocks_per_sm)

                best = None
                total = 0
                try:
                    for _ in range(max(1, args.repeat)):
                        elapsed, total = run_once(args.cycle_enum, cli_args, env)
                        best = elapsed if best is None else min(best, elapsed)
                except subprocess.CalledProcessError as error:
                    sys.stderr.write(
                        f"block={block_size} bpsm={blocks_per_sm} failed: "
                        f"{error.stderr}\n"
                    )
                    continue

                writer.writerow([block_size, blocks_per_sm, f"{best:.6f}", total])
                handle.flush()
                print(
                    f"block={block_size} blocks_per_sm={blocks_per_sm} "
                    f"best={best:.6f}s total={total}"
                )

    print(f"Tuning sweep written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
