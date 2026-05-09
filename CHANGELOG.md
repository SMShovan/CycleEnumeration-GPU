# Changelog

All notable project changes will be recorded in this file.

Entries should stay brief and practical: what changed, why it changed, and
which implementation or optimization concern it supports. The changelog is also
intended to preserve enough technical context for a later project report.

## [Unreleased]

### Added

- Added initial repository metadata and ignore rules for build output,
  generated documentation, benchmark results, CUDA profiling artifacts, and
  local-only planning notes.
- Added a README describing the staged implementation path: sequential
  reference code, OpenMP comparison baselines, and single-GPU CUDA
  implementations.
- Documented that the project is single-node only and will compare final
  performance against the TBB-based baseline paper implementation.
- Added a target-based CMake skeleton with centralized build options for tests,
  benchmarks, OpenMP, CUDA, Doxygen, and sanitizers.
- Added a small core library target with version helpers so future CLI and
  benchmark output can record the exact project build version.
- Added Google Test and CTest wiring with a smoke test for the compiled project
  version. This creates the validation path that later parser, algorithm, and
  CUDA work will use.
- Added optional Doxygen wiring and a project Doxyfile template. Public headers
  can now grow API documentation as the graph, CPU, OpenMP, and CUDA modules
  are implemented.

### Notes

- No cycle enumeration algorithm code has been added yet. These foundation
  steps keep the public repository clean before graph parsing, tests, and
  algorithm implementations are introduced.
