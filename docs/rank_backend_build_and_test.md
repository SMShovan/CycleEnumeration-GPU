# Rank blocking backend: build and test

Branch: `feature/cuda-rank-blocking`. New backend selected with `--blocking rank`,
with an optional, toggleable hub safety valve `--step-budget N` (0 = off = exact).
The non-blocking and certify backends are unchanged.

## Build (same toolchain as the other CUDA backends)

On a GPU node:

```sh
module load cuda/11.8.0_gcc_9.5.0 gcc/9.5.0
cd CycleEnumeration-GPU
rm -rf build-cuda
cmake -S . -B build-cuda \
  -DCYCLE_ENUM_BUILD_TESTS=ON \
  -DCYCLE_ENUM_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=90 \
  -DCMAKE_C_COMPILER=$(which gcc) \
  -DCMAKE_CXX_COMPILER=$(which g++) \
  -DCMAKE_CUDA_HOST_COMPILER=$(which g++)
cmake --build build-cuda --parallel 8
```

If `nvcc` reports an error from `cuda_johnson_rank_kernel.cu`, send it to me and I
will fix it; the host-side files already compile clean with g++ 11.

## Tests

```sh
ctest --test-dir build-cuda --output-on-failure -R "CudaRankTest"
```

Expected:
- `CudaRankTest.CounterexampleIsExactUnderCap`: with the budget off, the rank
  backend gets the cycle the certify backend misses (exact under the length cap).
- `CudaRankTest.ExactMatchesBruteForceOnDenseGraph`: rank equals non-blocking and
  brute force across several caps, zero flagged.
- `CudaRankTest.StepBudgetIsLowerBoundAndFlags`: with a tight budget the result is
  a lower bound and flags roots; with a huge budget it reproduces the exact count
  with nothing flagged.

Run the full suite too, to confirm the other backends still pass:

```sh
ctest --test-dir build-cuda --output-on-failure
```

## CLI

Exact run (no hubs, or you want the true count):

```sh
./build-cuda/cycle-enum --input tests/data/reference_sample.txt \
  --backend cuda --mode simple --algorithm johnson \
  --cuda-scheduler work-queue --max-cycle-length 6 \
  --blocking rank --report-timing
```

Hub safety valve on (cap each root's work; result becomes a certified lower bound):

```sh
./build-cuda/cycle-enum --input <big-graph> \
  --backend cuda --mode simple --algorithm johnson \
  --cuda-scheduler work-queue --max-cycle-length 6 \
  --blocking rank --step-budget 5000000 --report-uncertain flagged.txt
```

Footer on stdout (a `#` comment, ignored by the benchmark parser):
`# status: exact (0 of N roots hit the step budget)` when nothing was truncated,
or `# status: lower-bound (U of N roots hit the step budget)` otherwise. With
`--report-uncertain` the flagged root ids are written to the given file.

## Selecting among the three CUDA blocking backends

- `--blocking none` (default): existing non-blocking exact kernel.
- `--blocking certify`: lower-bound kernel with the length-cap certificate (P1).
- `--blocking rank`: this proposed method. Exact with `--step-budget 0`; a
  certified lower bound with `--step-budget N > 0`.

## Notes

- With the budget off, the only approximation source is gone: the rank makes the
  length cap exact. Uncertainty appears only when you turn the budget on.
- Memory: per thread this stores one 32-bit value per vertex (no |E|-sized
  structure), so it frees occupancy relative to the certify backend on graphs
  with average degree above roughly 8 to 16. On very large graphs the backend
  clamps the resident grid to fit memory and warns if it must shrink.
- The exact iterative logic was validated on CPU against brute force over 10,800
  random (graph, length) cases before porting to CUDA.
