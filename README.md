# CycleEnumeration-GPU

CycleEnumeration-GPU is a single-node C++/CUDA project for directed cycle
enumeration. The project will grow in stages: sequential reference
implementations first, OpenMP CPU baselines next, and then CUDA implementations
for one NVIDIA GPU.

The final performance comparison target is the TBB-based baseline from the
SPAA 2022 paper "Scalable Fine-Grained Parallel Cycle Enumeration Algorithms".
This repository does not use MPI or distributed execution; scalability work is
focused on CPU threading and one-GPU execution.

## Planned Implementation Path

1. Build a conventional CMake project with Google Test, CTest, and Doxygen.
2. Add graph parsing, CSR/CSC graph views, timestamp helpers, and histogram
   utilities.
3. Implement exact sequential cycle counters.
4. Add OpenMP CPU implementations for comparison.
5. Add naive CUDA implementations that prioritize correctness.
6. Optimize CUDA for large temporal graphs using GPU-friendly graph layout,
   dynamic work distribution, reduced atomic contention, timestamp lookup
   optimizations, and explicit device-side DFS state.
7. Add benchmark scripts for comparing sequential, OpenMP, CUDA, and the TBB
   baseline.

## Current Status

The repository has the foundation, temporal graph parser, CSR/CSC graph view,
histogram utilities, timestamp helpers, exact sequential counters, and OpenMP
CPU baselines. The `cycle-enum` command-line driver can run sequential and
OpenMP static simple-cycle counters plus temporal Johnson and Read-Tarjan
counters.

CUDA runtime detection is wired into the build as an optional target. On
non-CUDA machines it compiles as an unavailable backend; on NVIDIA systems,
`CYCLE_ENUM_ENABLE_CUDA=ON` enables CUDA language support and links the runtime.

## Build and Run

```sh
cmake -S . -B build -DCYCLE_ENUM_BUILD_TESTS=ON
cmake --build build
ctest --test-dir build --output-on-failure
```

CUDA configuration for an H100-class cluster can start with:

```sh
cmake -S . -B build-cuda \
  -DCYCLE_ENUM_BUILD_TESTS=ON \
  -DCYCLE_ENUM_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=90
cmake --build build-cuda
```

Example sequential run:

```sh
./build/cycle-enum --input tests/data/reference_sample.txt \
  --algorithm johnson \
  --mode simple-time-window \
  --time-window 3600
```

The output uses the baseline-compatible histogram format:

```text
# cycle_size, num_of_cycles
2, 1
3, 1
4, 2
5, 1
Total, 5
```

## Target Platform

Development starts on macOS, where CUDA kernels cannot be executed locally.
CUDA code will therefore be written to compile on NVIDIA systems and guarded so
non-CUDA builds remain usable. Final GPU validation and performance measurement
are intended for a Linux cluster with NVIDIA H100 GPUs.

## Input Format

The planned temporal graph format is one directed edge per line:

```text
source target timestamp
```

Lines beginning with `#` or `%` are treated as comments. Self-loops are ignored
by the planned parser.
