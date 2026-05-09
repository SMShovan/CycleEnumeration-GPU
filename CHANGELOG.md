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
- Added shared option and result types for selecting cycle semantics,
  algorithm family, execution backend, validation behavior, and runtime
  reporting. These types give the later sequential, OpenMP, and CUDA paths a
  common interface.
- Added unit tests for option defaults, option validation, status names, and
  runtime-statistic defaults.
- Added a deterministic cycle histogram utility with ordered output, merging,
  total-count reporting, and overflow checks. This will be the shared result
  container for correctness tests, CLI output, benchmark logs, and baseline
  comparisons.
- Added temporal graph parsing for `source target timestamp` input. The parser
  skips comments and self-loops, compacts external vertex ids, groups duplicate
  directed edges, preserves duplicate timestamps, and sorts timestamp lists.
- Added parser tests for comments, duplicate edge grouping, timestamp sorting,
  self-loop skipping, missing files, malformed rows, and vertex-id compaction.
- Updated vertex compaction to preserve ascending external-id order. This keeps
  duplicate-avoidance comparisons consistent with the numeric vertex ordering
  used by the baseline implementation.
- Added CSR and CSC graph views with flat timestamp storage. The outgoing view
  supports forward traversal, the incoming view supports reverse reachability
  and future cycle-union preprocessing, and shared timestamp ranges keep both
  views consistent for CPU and CUDA transfer planning.
- Added host timestamp interval helpers for inclusive time-window lookup,
  strictly-after-start lookup, and temporal increasing-order lookup. These
  helpers define the boundary semantics that CPU and CUDA algorithms will need
  to match.
- Added a tiny-graph brute-force simple-cycle oracle. It exhaustively searches
  directed cycles while counting each cycle once by requiring the root to be the
  smallest compact vertex in the cycle. This oracle is intended for correctness
  tests of Johnson, Read-Tarjan, OpenMP, and CUDA implementations.
- Added a sequential Johnson simple-cycle counter for static directed graphs.
  The implementation uses blocked-set and blocked-list pruning, counts each
  directed cycle once using the smallest compact vertex as root, and is tested
  against the brute-force oracle on acyclic, single-cycle, and overlapping-cycle
  fixtures.
- Added sequential Johnson and brute-force oracle support for simple cycles
  constrained by start-edge time windows. The timestamp boundary rule follows
  the baseline implementation's vertex-order convention, and the reference
  sample graph now reproduces the expected histogram of five cycles.
- Added a sequential Read-Tarjan-style static cycle counter based on
  path-extension search. The implementation is validated against the brute-force
  oracle on acyclic, single-cycle, overlapping-cycle, and maximum-length
  fixtures.
- Added sequential Read-Tarjan support for start-edge time-window counting.
  The implementation shares the baseline-compatible timestamp convention used
  by the Johnson time-window path and is checked against the brute-force oracle.
- Added the first command-line driver for sequential counting. It reads the
  temporal graph format, dispatches Johnson, Read-Tarjan, or brute-force
  counters for static and simple time-window modes, and emits the stable
  histogram format used for later baseline comparisons.
- Added a brute-force oracle for temporal cycles. It counts timestamped cycle
  instances by requiring strictly increasing edge timestamps within the start
  edge's window, which gives later optimized temporal implementations a small
  but exact validator.
- Added a naive sequential temporal Johnson counter. This version keeps a
  per-start-edge timestamp DFS with explicit visited/path state and validates
  against the temporal brute-force oracle before later closing-time pruning.
- Added conservative closing-time pruning to sequential temporal Johnson. The
  pruning records timestamp thresholds for vertices that cannot temporally
  reach the root inside a fixed start window, and exposes lightweight counters
  so tests and benchmark reports can show when pruning occurs.

### Notes

- No cycle enumeration algorithm code has been added yet. These foundation
  steps keep the public repository clean before graph parsing, tests, and
  algorithm implementations are introduced.
