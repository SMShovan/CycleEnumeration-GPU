# Blocking backend: build and test on the H100 node

Branch: `feature/cuda-blocking-backend`. Adds `--blocking yes|no` to the CUDA
static simple-cycle work-queue path, with a per-root uncertainty certificate.

The blocking backend cannot be compiled without `nvcc`, so build and test on a
GPU node (interactive Slurm allocation), not the login node.

## 1. Load the toolchain

CUDA 11.8 caps the host compiler at GCC 11, and the cluster pairs it with GCC
9.5.0, so use that exact compiler (the same setup as the working baseline build).

```sh
module load cuda/11.8.0_gcc_9.5.0 gcc/9.5.0
which gcc g++ ; g++ --version | head -1     # expect GCC 9.5.0
```

## 2. Configure (clean) with the host compiler pinned

```sh
cd CycleEnumeration-GPU
rm -rf build-cuda
cmake -S . -B build-cuda \
  -DCYCLE_ENUM_BUILD_TESTS=ON \
  -DCYCLE_ENUM_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=90 \
  -DCMAKE_C_COMPILER=$(which gcc) \
  -DCMAKE_CXX_COMPILER=$(which g++) \
  -DCMAKE_CUDA_HOST_COMPILER=$(which g++)
```

## 3. Build

```sh
cmake --build build-cuda --parallel 8
```

If `nvcc` reports `unsupported GNU version`, the wrong GCC is loaded; reload
`gcc/9.5.0`. If a build error comes from `cuda_johnson_blocked_kernel.cu`, send
me the message and I will fix it.

## 4. Run the new tests

The host-only cross-map test runs anywhere; the GPU tests self-skip when no
device is visible, so run them inside the allocation.

```sh
ctest --test-dir build-cuda --output-on-failure -R "CudaGraphTest|CudaBlockingTest"
```

Expected:
- `CudaGraphTest.BuildsCscToCsrCrossMapForBlocking` passes (P0 cross-map).
- `CudaBlockingTest.NoCapMatchesNonBlockingAndBruteForce` passes: with a large
  cap, blocking == non-blocking == brute force and zero roots are flagged.
- `CudaBlockingTest.CounterexampleUndercountsAndFlagsRoot` passes: with a tight
  cap, blocking is a strict lower bound and flags the affected root.

Run the full suite too, to confirm nothing else regressed:

```sh
ctest --test-dir build-cuda --output-on-failure
```

## 5. CLI smoke test

```sh
# Blocking on a small graph: with max length 6 the cap should not bite, so the
# footer should read "exact" and match the non-blocking counts.
./build-cuda/cycle-enum --input tests/data/reference_sample.txt \
  --backend cuda --mode simple --algorithm johnson \
  --cuda-scheduler work-queue --max-cycle-length 6 \
  --blocking yes --report-timing

# Same run without blocking, to compare the histogram directly.
./build-cuda/cycle-enum --input tests/data/reference_sample.txt \
  --backend cuda --mode simple --algorithm johnson \
  --cuda-scheduler work-queue --max-cycle-length 6 --blocking no
```

The blocking run prints the histogram, then a footer comment line on stdout, for
example `# status: exact (0 of N roots hit the length cap)`, and
`flagged_roots: 0` on stderr. The benchmark parser ignores `#` lines, so adding
`--blocking yes` does not break `run_dataset_benchmarks.py`. Add
`--report-uncertain flagged.txt` to dump the flagged root ids.

## 6. What to expect on a real dataset

On a hub-heavy graph at a tight cap, expect many roots to be flagged (the cap
bites often), so the blocking histogram is reported as a lower bound. On sparse
graphs or with a generous cap, expect zero flags and a certified-exact result.
If the graph is large enough that per-thread blocked/B buffers do not fit, the
backend reduces the resident grid; if even one block does not fit it errors and
asks for a smaller `--max-cycle-length` or `--blocking no`.
