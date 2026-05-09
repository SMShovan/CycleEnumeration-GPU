#!/usr/bin/env python3
"""Run repeatable cycle-enumeration benchmark sweeps.

The harness intentionally depends only on the Python standard library.  That
keeps it usable on a macOS development machine, a cluster login node, or an
allocated GPU node without asking the environment for extra packages.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import pathlib
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Iterable


CSV_FIELDS = [
    "suite_id",
    "utc_start",
    "input",
    "runner",
    "backend",
    "algorithm",
    "mode",
    "threads",
    "cuda_device",
    "tbb_algo",
    "time_window_seconds",
    "max_cycle_length",
    "repeat_index",
    "elapsed_ms",
    "reported_total_seconds",
    "return_code",
    "status",
    "total_cycles",
    "histogram",
    "vertex_visits",
    "command",
    "stderr_tail",
]


TBB_ALGORITHMS = {
    0: ("johnson-time-window-coarse", "simple-time-window"),
    1: ("johnson-time-window-fine", "simple-time-window"),
    2: ("read-tarjan-time-window-coarse", "simple-time-window"),
    3: ("read-tarjan-time-window-fine", "simple-time-window"),
    4: ("temporal-johnson-coarse", "temporal"),
    5: ("temporal-johnson-fine", "temporal"),
    6: ("temporal-read-tarjan-coarse", "temporal"),
    7: ("temporal-read-tarjan-fine", "temporal"),
    8: ("temporal-read-tarjan-single", "temporal"),
    9: ("2scent-no-source", "temporal"),
    10: ("2scent-source", "temporal"),
    11: ("cycle-union-time", "preprocessing"),
}


@dataclass(frozen=True)
class BenchmarkCase:
    """A concrete command with enough metadata for one CSV row."""

    input_path: pathlib.Path
    runner: str
    backend: str
    algorithm: str
    mode: str
    command: list[str]
    threads: int | None = None
    cuda_device: int | None = None
    tbb_algo: int | None = None


@dataclass(frozen=True)
class CommandResult:
    """Normalized process result used by the CSV writer and exit logic."""

    elapsed_ms: float
    return_code: int | None
    status: str
    stdout: str
    stderr: str


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def max_cycle_length(value: str) -> int:
    parsed = positive_int(value)
    if parsed < 2:
        raise argparse.ArgumentTypeError("max cycle length must be at least 2")
    return parsed


def default_cycle_enum_path(build_dir: pathlib.Path) -> pathlib.Path:
    return build_dir / "cycle-enum"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark CycleEnumeration-GPU backends and, optionally, the "
            "SPAA/TBB baseline executable."
        )
    )
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        type=pathlib.Path,
        help="Temporal graph input path. May be provided more than once.",
    )
    parser.add_argument(
        "--build-dir",
        type=pathlib.Path,
        default=pathlib.Path("build"),
        help="CMake build directory used to locate cycle-enum by default.",
    )
    parser.add_argument(
        "--cycle-enum",
        type=pathlib.Path,
        help="Path to the CycleEnumeration-GPU CLI executable.",
    )
    parser.add_argument(
        "--backend",
        action="append",
        choices=["sequential", "openmp", "cuda"],
        help="Project backend to benchmark. May be repeated.",
    )
    parser.add_argument(
        "--algorithm",
        action="append",
        choices=["johnson", "read-tarjan", "brute-force"],
        help="Project algorithm to benchmark. May be repeated.",
    )
    parser.add_argument(
        "--mode",
        action="append",
        choices=["simple", "simple-time-window", "temporal"],
        help="Cycle semantics to benchmark. May be repeated.",
    )
    parser.add_argument(
        "--openmp-threads",
        action="append",
        type=positive_int,
        help="OpenMP thread count. May be repeated.",
    )
    parser.add_argument(
        "--cuda-device",
        action="append",
        type=non_negative_int,
        help="CUDA device id. May be repeated.",
    )
    parser.add_argument(
        "--time-window",
        type=positive_int,
        default=3600,
        help="Time window in seconds for time-window and temporal modes.",
    )
    parser.add_argument(
        "--max-cycle-length",
        type=max_cycle_length,
        help="Optional maximum cycle length; required for the CUDA backend.",
    )
    parser.add_argument(
        "--repeat",
        type=positive_int,
        default=3,
        help="Number of measured repetitions for each case.",
    )
    parser.add_argument(
        "--warmup",
        type=non_negative_int,
        default=1,
        help="Warmup repetitions before measured runs.",
    )
    parser.add_argument(
        "--timeout",
        type=positive_int,
        help="Per-run timeout in seconds.",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path("benchmarks/results/cycle_benchmarks.csv"),
        help="CSV output path.",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to an existing CSV instead of replacing it.",
    )
    parser.add_argument(
        "--allow-failures",
        action="store_true",
        help="Return success even if one or more measured commands fail.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop the sweep after the first failing command.",
    )
    parser.add_argument(
        "--tbb-exe",
        type=pathlib.Path,
        help="Optional path to the SPAA/TBB baseline cycle executable.",
    )
    parser.add_argument(
        "--tbb-algo",
        action="append",
        type=non_negative_int,
        help="TBB baseline algorithm id. May be repeated.",
    )
    parser.add_argument(
        "--tbb-threads",
        action="append",
        type=positive_int,
        help="TBB baseline thread count. May be repeated.",
    )
    parser.add_argument(
        "--tbb-use-cycle-union",
        action="store_true",
        help="Pass -cunion to the TBB baseline command.",
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    cycle_enum = args.cycle_enum or default_cycle_enum_path(args.build_dir)
    if not cycle_enum.exists():
        raise SystemExit(f"cycle-enum executable not found: {cycle_enum}")

    for input_path in args.input:
        if not input_path.exists():
            raise SystemExit(f"input file not found: {input_path}")

    if args.tbb_exe is not None and not args.tbb_exe.exists():
        raise SystemExit(f"TBB baseline executable not found: {args.tbb_exe}")

    if "cuda" in (args.backend or ["sequential"]) and args.max_cycle_length is None:
        raise SystemExit("CUDA benchmark cases require --max-cycle-length")


def project_cases(args: argparse.Namespace) -> Iterable[BenchmarkCase]:
    cycle_enum = str(args.cycle_enum or default_cycle_enum_path(args.build_dir))
    backends = args.backend or ["sequential"]
    algorithms = args.algorithm or ["johnson"]
    modes = args.mode or ["simple"]
    openmp_threads = args.openmp_threads or [1]
    cuda_devices = args.cuda_device or [0]

    for input_path in args.input:
        for backend in backends:
            for algorithm in algorithms:
                for mode in modes:
                    if backend == "openmp" and algorithm == "brute-force":
                        print(
                            "Skipping unsupported OpenMP brute-force case",
                            file=sys.stderr,
                        )
                        continue
                    if backend == "openmp" and mode == "simple-time-window":
                        print(
                            "Skipping unsupported OpenMP simple-time-window case",
                            file=sys.stderr,
                        )
                        continue
                    if backend == "cuda" and (
                        algorithm != "johnson"
                        or mode not in ("simple", "simple-time-window")
                    ):
                        print(
                            "Skipping unsupported CUDA case "
                            f"algorithm={algorithm} mode={mode}",
                            file=sys.stderr,
                        )
                        continue

                    thread_values = openmp_threads if backend == "openmp" else [None]
                    device_values = cuda_devices if backend == "cuda" else [None]
                    for threads in thread_values:
                        for cuda_device in device_values:
                            command = [
                                cycle_enum,
                                "--input",
                                str(input_path),
                                "--backend",
                                backend,
                                "--algorithm",
                                algorithm,
                                "--mode",
                                mode,
                            ]
                            if mode != "simple":
                                command.extend(["--time-window", str(args.time_window)])
                            if args.max_cycle_length is not None:
                                command.extend(
                                    ["--max-cycle-length", str(args.max_cycle_length)]
                                )
                            if threads is not None:
                                command.extend(["--openmp-threads", str(threads)])
                            if cuda_device is not None:
                                command.extend(["--cuda-device", str(cuda_device)])

                            yield BenchmarkCase(
                                input_path=input_path,
                                runner="cycle-enum",
                                backend=backend,
                                algorithm=algorithm,
                                mode=mode,
                                command=command,
                                threads=threads,
                                cuda_device=cuda_device,
                            )


def tbb_cases(args: argparse.Namespace) -> Iterable[BenchmarkCase]:
    if args.tbb_exe is None:
        return

    algorithms = args.tbb_algo or [0]
    threads_values = args.tbb_threads or args.openmp_threads or [1]
    for input_path in args.input:
        for algo in algorithms:
            algorithm_name, mode = TBB_ALGORITHMS.get(
                algo, (f"algo-{algo}", "baseline")
            )
            for threads in threads_values:
                command = [
                    str(args.tbb_exe),
                    "-f",
                    str(input_path),
                    "-algo",
                    str(algo),
                    "-tws",
                    str(args.time_window),
                    "-n",
                    str(threads),
                ]
                if args.tbb_use_cycle_union:
                    command.append("-cunion")

                yield BenchmarkCase(
                    input_path=input_path,
                    runner="tbb-baseline",
                    backend="tbb",
                    algorithm=algorithm_name,
                    mode=mode,
                    command=command,
                    threads=threads,
                    tbb_algo=algo,
                )


def run_command(command: list[str], timeout: int | None) -> CommandResult:
    start = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        status = "ok" if completed.returncode == 0 else "failed"
        return CommandResult(
            elapsed_ms=elapsed_ms,
            return_code=completed.returncode,
            status=status,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )
    except subprocess.TimeoutExpired as error:
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        stdout = error.stdout if isinstance(error.stdout, str) else ""
        stderr = error.stderr if isinstance(error.stderr, str) else ""
        return CommandResult(
            elapsed_ms=elapsed_ms,
            return_code=None,
            status="timeout",
            stdout=stdout,
            stderr=stderr,
        )


def parse_histogram(stdout: str) -> tuple[int | None, str]:
    bins: dict[int, int] = {}
    total: int | None = None
    for raw_line in stdout.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "," not in line:
            continue
        left, right = [part.strip() for part in line.split(",", 1)]
        try:
            if left.lower() == "total":
                total = int(right)
            elif left.isdigit():
                bins[int(left)] = int(right)
        except ValueError:
            continue
    histogram = ";".join(f"{length}:{count}" for length, count in sorted(bins.items()))
    return total, histogram


def parse_reported_seconds(stdout: str) -> float | None:
    match = re.search(r"Total time:\s*([0-9]+(?:\.[0-9]+)?)\s*s", stdout)
    return float(match.group(1)) if match else None


def parse_vertex_visits(stdout: str) -> int | None:
    match = re.search(r"Vertex visits:\s*([0-9]+)", stdout)
    return int(match.group(1)) if match else None


def collapsed_tail(text: str, max_chars: int = 400) -> str:
    tail = text[-max_chars:]
    return " ".join(tail.split())


def row_from_result(
    suite_id: str,
    utc_start: str,
    case: BenchmarkCase,
    repeat_index: int,
    result: CommandResult,
    args: argparse.Namespace,
) -> dict[str, object]:
    total_cycles, histogram = parse_histogram(result.stdout)
    reported_seconds = parse_reported_seconds(result.stdout)
    vertex_visits = parse_vertex_visits(result.stdout)
    status = result.status
    if status == "ok" and total_cycles is None:
        status = "missing-total"

    return {
        "suite_id": suite_id,
        "utc_start": utc_start,
        "input": str(case.input_path),
        "runner": case.runner,
        "backend": case.backend,
        "algorithm": case.algorithm,
        "mode": case.mode,
        "threads": "" if case.threads is None else case.threads,
        "cuda_device": "" if case.cuda_device is None else case.cuda_device,
        "tbb_algo": "" if case.tbb_algo is None else case.tbb_algo,
        "time_window_seconds": args.time_window,
        "max_cycle_length": "" if args.max_cycle_length is None else args.max_cycle_length,
        "repeat_index": repeat_index,
        "elapsed_ms": f"{result.elapsed_ms:.3f}",
        "reported_total_seconds": (
            "" if reported_seconds is None else reported_seconds
        ),
        "return_code": "" if result.return_code is None else result.return_code,
        "status": status,
        "total_cycles": "" if total_cycles is None else total_cycles,
        "histogram": histogram,
        "vertex_visits": "" if vertex_visits is None else vertex_visits,
        "command": shlex.join(case.command),
        "stderr_tail": collapsed_tail(result.stderr),
    }


def open_writer(path: pathlib.Path, append: bool) -> tuple[object, csv.DictWriter]:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    file_obj = path.open(mode, newline="")
    writer = csv.DictWriter(file_obj, fieldnames=CSV_FIELDS)
    if not append or path.stat().st_size == 0:
        writer.writeheader()
    return file_obj, writer


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_args(args)

    suite_id = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    cases = list(project_cases(args)) + list(tbb_cases(args))
    if not cases:
        print("No benchmark cases selected.", file=sys.stderr)
        return 1

    failures = 0
    measured_rows = 0
    output_file, writer = open_writer(args.output, args.append)
    with output_file:
        for case in cases:
            for _ in range(args.warmup):
                warmup_result = run_command(case.command, args.timeout)
                if warmup_result.status != "ok":
                    failures += 1
                    print(
                        "Warmup failed for "
                        f"{case.runner} {case.backend} {case.algorithm}: "
                        f"{collapsed_tail(warmup_result.stderr)}",
                        file=sys.stderr,
                    )
                    if args.fail_fast:
                        return 1

            for repeat_index in range(args.repeat):
                utc_start = dt.datetime.now(dt.timezone.utc).isoformat()
                result = run_command(case.command, args.timeout)
                row = row_from_result(
                    suite_id=suite_id,
                    utc_start=utc_start,
                    case=case,
                    repeat_index=repeat_index,
                    result=result,
                    args=args,
                )
                writer.writerow(row)
                output_file.flush()
                measured_rows += 1

                if row["status"] != "ok":
                    failures += 1
                    print(
                        "Benchmark failed for "
                        f"{case.runner} {case.backend} {case.algorithm}: "
                        f"{row['status']}",
                        file=sys.stderr,
                    )
                    if args.fail_fast:
                        return 1

    print(f"Wrote {measured_rows} benchmark rows to {args.output}")
    if failures and not args.allow_failures:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
