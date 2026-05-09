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

### Notes

- No implementation code has been added yet. This foundation step keeps the
  public repository clean before CMake, tests, and algorithms are introduced.
