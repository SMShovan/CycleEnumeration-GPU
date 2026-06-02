# CUDA Launch Tuning

This note describes how the CUDA work-queue launch is tuned and how the chosen
defaults are recorded.

## Tunable parameters

The persistent work-queue kernel reads two launch parameters from the
environment so a sweep can vary them without rebuilding:

- `CYCLE_ENUM_CUDA_BLOCK_SIZE` — threads per block, default 128. Must be a
  positive multiple of the 32-thread warp size.
- `CYCLE_ENUM_CUDA_BLOCKS_PER_SM` — resident blocks per multiprocessor, default
  16. Together with the device multiprocessor count this sets the resident grid,
  which `plan_work_queue_launch` caps at the work-item count.

`work_queue_tuning_from_env` reads and validates these values. An unset variable
keeps the default; a present but invalid value (non-positive, unparseable, or a
block size that is not a multiple of 32) is rejected so sweep typos fail loudly
rather than silently running a default. This reader is unit tested.

Stack capacity (`max_cycle_length`) and the histogram aggregation strategy
(per-block shared memory, flushed to global) are also part of the launch
configuration. The stack capacity is a correctness bound chosen by the caller;
the histogram strategy is fixed for now and listed here so the sweep can be
extended if a second strategy is added.

## Sweep

`scripts/tune_cuda.py` sweeps block size and blocks-per-SM, runs the cycle-enum
CLI for each combination with the matching environment, and records the best
wall-clock time and the total cycle count per combination in a CSV:

```sh
python3 benchmarks/scripts/tune_cuda.py \
  --cycle-enum build-h100/cycle-enum \
  --input /path/to/graph.txt \
  --mode temporal \
  --max-cycle-length 8 \
  --block-sizes 64 128 256 512 \
  --blocks-per-sm 8 16 32 \
  --repeat 5 \
  --output benchmarks/results/tune-temporal.csv
```

The total cycle count column is a guard: every combination must report the same
total, since only scheduling changes. A differing total signals a launch
configuration that breaks correctness and must be investigated before trusting
its timing.

## Recording defaults

After a sweep on representative datasets, the block size and blocks-per-SM with
the best stable time become the documented defaults. If the data shows one
combination is consistently best, the defaults in `work_queue_tuning_from_env`
are updated to match; otherwise the environment variables let each dataset pick
its own. The chosen defaults and the dataset they were tuned on are recorded
alongside the benchmark results.
