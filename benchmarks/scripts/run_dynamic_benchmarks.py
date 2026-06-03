#!/usr/bin/env python3
"""Sweep incremental-update vs full-recompute timings for the dynamic backend.

Drives the cycle-enum CLI in update mode with --compare-recompute across a sweep
of batch sizes, locality windows, and seeds, parsing the reported update and
recompute timings and the correctness match. Writes one CSV row per case with the
speedup of update over recompute, so the recompute-versus-update crossover can be
plotted against batch size and locality.

The prior histogram is computed inside the CLI but is not part of the timed
update (it is cached prior knowledge), so the recorded update time measures only
the incremental work. Dependency-free for cluster use.
"""

import argparse
import csv
import os
import re
import subprocess
import sys

UPDATE_RE = re.compile(r"update_seconds=([0-9.eE+-]+)")
RECOMPUTE_RE = re.compile(r"recompute_seconds=([0-9.eE+-]+)")
MATCH_RE = re.compile(r"match=(\w+)")
COUNTS_RE = re.compile(r"deletions=(\d+)\s+insertions=(\d+)")
TOTAL_RE = re.compile(r"Total,\s*(\d+)")


def run_case(cli, base_args, deletes, inserts, locality, seed):
    args = [
        cli, *base_args,
        "--task", "update",
        "--compare-recompute",
        "--deletes", str(deletes),
        "--inserts", str(inserts),
        "--batch-seed", str(seed),
    ]
    if locality is not None:
        args += ["--batch-locality", str(locality)]
    proc = subprocess.run(args, capture_output=True, text=True, check=True)
    info = proc.stderr + proc.stdout
    update = float(UPDATE_RE.search(info).group(1))
    recompute = float(RECOMPUTE_RE.search(info).group(1))
    match = MATCH_RE.search(info).group(1)
    counts = COUNTS_RE.search(info)
    total = TOTAL_RE.search(proc.stdout)
    return {
        "deletions": counts.group(1) if counts else "",
        "insertions": counts.group(2) if counts else "",
        "update_seconds": update,
        "recompute_seconds": recompute,
        "match": match,
        "total_cycles": total.group(1) if total else "",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cycle-enum", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--backend", default="sequential",
                        choices=["sequential", "openmp", "cuda"])
    parser.add_argument("--max-cycle-length", type=int, required=True)
    parser.add_argument("--openmp-threads", type=int, default=1)
    parser.add_argument("--cuda-device", type=int, default=0)
    parser.add_argument("--deletes", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--inserts", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--locality", type=int, nargs="*", default=[],
                        help="locality windows; empty means global changes")
    parser.add_argument("--seeds", type=int, default=5,
                        help="number of seeds per configuration")
    parser.add_argument("--repeat", type=int, default=3,
                        help="measured repeats per case (min update time kept)")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    base_args = [
        "--input", args.input,
        "--backend", args.backend,
        "--mode", "simple",
        "--max-cycle-length", str(args.max_cycle_length),
        "--openmp-threads", str(args.openmp_threads),
        "--cuda-device", str(args.cuda_device),
    ]
    localities = args.locality if args.locality else [None]

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "backend", "threads", "device", "deletes", "inserts", "locality",
            "seed", "deletions", "insertions", "update_seconds",
            "recompute_seconds", "speedup", "match", "total_cycles",
        ])
        for deletes in args.deletes:
            for inserts in args.inserts:
                for locality in localities:
                    for seed in range(args.seeds):
                        best = None
                        try:
                            for _ in range(max(1, args.repeat)):
                                result = run_case(args.cycle_enum, base_args,
                                                  deletes, inserts, locality,
                                                  seed)
                                if best is None or \
                                        result["update_seconds"] < best["update_seconds"]:
                                    best = result
                        except subprocess.CalledProcessError as error:
                            sys.stderr.write(
                                f"case d={deletes} i={inserts} loc={locality} "
                                f"seed={seed} failed: {error.stderr}\n")
                            continue
                        speedup = (best["recompute_seconds"] /
                                   best["update_seconds"]
                                   if best["update_seconds"] > 0 else "")
                        writer.writerow([
                            args.backend, args.openmp_threads, args.cuda_device,
                            deletes, inserts,
                            locality if locality is not None else "",
                            seed, best["deletions"], best["insertions"],
                            f"{best['update_seconds']:.9f}",
                            f"{best['recompute_seconds']:.9f}",
                            f"{speedup:.3f}" if speedup != "" else "",
                            best["match"], best["total_cycles"],
                        ])
                        handle.flush()

    print(f"Dynamic benchmark written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
