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
- Added timestamp path bundling to sequential temporal Johnson. The search now
  carries `(arrival timestamp, multiplicity)` bundles per vertex path, reducing
  recursive branching over timestamp assignments while preserving duplicate
  event multiplicity.
- Added the sequential temporal Read-Tarjan baseline. It uses independent
  start-edge timestamp work items with local visited/path state, conservative
  temporal reachability pruning, and parity tests against both the temporal
  brute-force oracle and temporal Johnson.
- Added sequential temporal cycle-union preprocessing. The first version
  computes a conservative forward/reverse reachability intersection inside a
  start edge's window, with tests for dead-branch removal, window handling, and
  invalid requests.
- Made CSR/CSC adjacency order deterministic for programmatically constructed
  graphs by sorting each vertex's adjacency entries by neighbor. This keeps
  path-extension algorithms consistent with parser-built graphs.
- Added a sequential validation matrix and validation design note. The matrix
  compares brute-force, Johnson, and Read-Tarjan implementations across static,
  simple time-window, and temporal modes so future OpenMP/CUDA work has a
  stable correctness gate.
- Added OpenMP build detection and runtime configuration helpers. The new
  `cycle_enum::openmp` target reports availability, max thread count, and
  validates requested thread counts while non-OpenMP builds remain usable.
- Added the first OpenMP Johnson baseline for static simple cycles. The
  implementation splits work by root vertex, uses thread-local histograms, and
  falls back to a single-thread path when OpenMP is unavailable.
- Added the OpenMP Read-Tarjan static baseline using the same root-level work
  partitioning and thread-local histogram merge strategy, with parity tests
  against the sequential Read-Tarjan counter.
- Added an OpenMP temporal Johnson baseline that parallelizes by root vertex
  while keeping DFS path state, closing-time pruning caches, and timestamp
  bundles private to each worker. The merged instrumentation mirrors the
  sequential temporal Johnson counters so CPU backend behavior can be compared
  in later benchmark reports.
- Added an OpenMP temporal Read-Tarjan baseline. It keeps the sequential
  timestamp-by-timestamp expansion semantics, adds thread-local instrumentation
  for DFS states, closing-time prunes, and timestamp extensions, and merges
  histograms after the root-parallel search.
- Extended the CLI to select `--backend sequential|openmp`, set
  `--openmp-threads`, and run sequential or OpenMP simple/temporal Johnson and
  Read-Tarjan counters. Unsupported backend and mode combinations now fail with
  explicit messages so benchmark scripts do not silently use the wrong backend.
- Added the initial CUDA build scaffold. The project now always exposes a
  `cycle_enum::cuda` target with host-side availability queries; non-CUDA
  machines compile a clear unavailable backend, while CUDA builds enable the
  CUDA language, link `cudart`, and default to the H100-oriented `sm_90`
  architecture unless overridden.
- Added a host-side CUDA graph packer that converts the shared CSR/CSC
  `GraphView` into fixed-width arrays for later device allocation. The layout
  keeps 32-bit vertex/edge ids but uses 64-bit offsets so timestamp-heavy
  temporal graphs remain representable on the GPU path.
- Added the first naive CUDA static Johnson counter. The CUDA build compiles a
  one-root-per-thread kernel with a bounded per-thread DFS stack and a global
  cycle-length histogram; non-CUDA builds expose the same API but fail clearly
  before dispatch. This is a correctness-oriented GPU baseline for later work
  on work queues, occupancy, and reduced atomic contention.
- Exposed the naive CUDA Johnson backend through the CLI for benchmark scripts.
  The CLI now accepts `--backend cuda` and `--cuda-device`; CUDA dispatch is
  intentionally limited to static Johnson with an explicit `--max-cycle-length`
  until the device stack and work-queue designs are generalized.
- Added a dependency-free benchmark harness under `benchmarks/`. It records
  repeatable CSV measurements for sequential, OpenMP, CUDA, and optional
  SPAA/TBB baseline runs, preserving histograms and command lines so later
  cluster experiments can connect performance numbers back to exact settings.
- Added CUDA host/device timestamp search helpers. These binary-search
  primitives mirror the CPU inclusive-window, strict-after-start, and temporal
  strictly-increasing timestamp semantics that future GPU time-window and
  temporal kernels will need.
- Added a naive CUDA Johnson simple-time-window counter. The kernel launches
  one thread per start-edge timestamp, performs bounded DFS with per-thread
  stacks, and uses device timestamp binary search to match the CPU
  inclusive/strict baseline boundary convention before later load-balancing
  and aggregation optimizations.
- Added a naive CUDA temporal Johnson counter. It keeps per-thread DFS frames
  for active edges and timestamp candidates so each valid timestamp assignment
  is counted explicitly, providing a correctness-first GPU temporal baseline
  before path bundling, pruning, and dynamic work distribution are introduced.
- Added per-block shared-memory histogram aggregation to CUDA Johnson kernels.
  Threads now accumulate cycle-length counts inside a block and flush one
  value per length to global memory, reducing global atomic contention for
  cycle-heavy graphs.
- Added an H100 cluster smoke script. It configures CUDA/OpenMP for `sm_90`,
  builds, runs CTest, and records a small sequential/OpenMP/CUDA Johnson
  benchmark CSV so cluster bring-up has a repeatable first command.
- Added a fine-grained OpenMP Read-Tarjan task experiment. It spawns one task
  per path prefix up to a configurable cutoff depth, copies path and visited
  state into each task in a copy-on-steal style, and finishes each subtree
  serially. The variant is a measurable comparison point against the
  coarse-grained baseline, not the primary CPU implementation, and is validated
  for histogram parity against the sequential and coarse-grained Read-Tarjan
  counters across thread counts and cutoff depths.
- Added an OpenMP CPU scaling benchmark script. It configures a CPU-only OpenMP
  build and drives the benchmark harness across a thread-count sweep for the
  sequential and OpenMP backends, so per-algorithm runtime and thread scaling
  can be recorded in one CSV without CUDA.
- Added a host-side CUDA start-event generator and temporal cycle-union
  prefilter. Start events are built once on the host, and the temporal dispatch
  drops start edges whose cycle-union set is empty before launching the kernel.
  The prefilter is correctness preserving and unit tested against an independent
  temporal-reachability reference, so it never removes a start edge that can
  close a cycle. The naive time-window and temporal kernels now consume the
  host-built events instead of regenerating them on the device.
- Added a CUDA temporal path-bundling investigation note. It evaluates on-GPU,
  host-only, and hybrid bundling and selects a hybrid that groups duplicate edge
  timestamps on the host and carries a device-side multiplicity scalar, deferring
  full path-dependent bundling until profiling justifies its unbounded per-thread
  state. The decision is recorded for the Phase 8 timestamp-lookup work.
- Optimized the CUDA graph memory layout by emitting the hot outgoing adjacency
  as a structure of arrays (split neighbor, timestamp-begin, and timestamp-end
  arrays) alongside the existing array-of-structs. The traversal kernels now read
  the split arrays through a shared device view and a centralized upload helper,
  which keeps warp loads coalesced and lets the static path skip timestamp data
  entirely. Host packing is unit tested to mirror the array-of-structs layout.
- Added a persistent work-queue CUDA path for static cycle counting. A fixed
  wave of resident blocks pulls root vertices from a global atomic counter
  instead of mapping one root to one thread, so skewed search trees no longer
  leave most threads idle. The host-side launch planner that sizes the resident
  grid to the device, capped by the work-item count, is unit tested, and the
  unavailable-backend and argument-validation paths are checked locally; device
  parity with the naive static counter is validated on the cluster.
- Added host-side CUDA branch splitting. It decomposes the static search into
  many short path-prefix work items so a few heavy roots no longer dominate the
  device, counting cycles shorter than the cutoff on the host and emitting the
  rest as continuation work items. The decomposition is exact: a unit test models
  the device continuation and confirms the recombined histogram equals the
  sequential Johnson count across cutoff depths and a bounded-length case.
- Added CUDA visited-set primitives. A header-only bitset over a caller-provided
  word array carries host and device qualifiers so it runs in registers or
  shared memory on the GPU and is unit tested on the host, alongside a mode
  selector that picks the bitset for graphs whose word array fits a tunable
  threshold and the sparse path scan otherwise. Bit operations are tested across
  word boundaries and for clear-all and mode selection.
- Added host-side CUDA timestamp grouping. Each outgoing edge's sorted timestamp
  list is collapsed into distinct values with multiplicities in a CSR-style
  layout, so temporal kernels iterate distinct times and multiply by the group
  count instead of branching over duplicate events. This implements the host half
  of the hybrid bundling decision and is unit tested for count preservation,
  per-edge ordering, and offset consistency.
- Documented the CUDA cycle-union placement decision. Host preprocessing is kept
  because it is already validated, overlaps with device execution, and has no
  measured bottleneck; the note records the concrete profiling criteria that
  would justify moving the reachability pass onto the GPU.
- Added NVTX profiling hooks. A header-only scoped-range guard brackets kernel
  and reduction phases and compiles as a no-op unless the build links NVTX, which
  CMake enables automatically on CUDA builds where the NVTX target is present. An
  Nsight profiling script runs a configuration under Nsight Systems and optionally
  Nsight Compute, and the no-op path is unit tested so the instrumentation never
  breaks the non-CUDA build.
- Made the CUDA work-queue launch tunable from the environment. The kernel reads
  a validated block size and resident blocks-per-multiprocessor from
  `CYCLE_ENUM_CUDA_BLOCK_SIZE` and `CYCLE_ENUM_CUDA_BLOCKS_PER_SM`, a sweep script
  records the best timing per combination with a total-cycle correctness guard,
  and the reader's defaults, overrides, and validation are unit tested.
- Promoted the persistent work-queue kernel to the default static CUDA path and
  exposed `--cuda-scheduler <naive|work-queue>` in the CLI, keeping the naive
  one-root-per-thread kernel available for debugging. Added a CUDA strategy
  design note, documented the scheduler in the README, and covered scheduler
  parsing and the unavailable-backend path with CLI tests.
- Documented the benchmark datasets. A datasets manifest records the temporal
  edge-list format, the in-repository fixtures, and the rules for external
  graphs, and a git-ignored `benchmarks/data/` directory holds local datasets so
  large or licensed graphs never enter version control.
- Added benchmark collection scripts. A matrix runner validates that every
  dataset exists before sweeping backends, algorithms, modes, and thread counts
  into one CSV, and an aggregation script groups the measured rows per
  configuration and reports run count, min, median, and mean elapsed time with a
  total-cycle column. Both were exercised against real harness output.
- Added a benchmark analysis script. It derives speedup relative to the
  single-thread sequential baseline of the same input, algorithm, and mode, and
  writes runtime bar charts when matplotlib is available while always producing
  the speedup table so it works on a bare cluster allocation. The benchmark
  README documents the matrix runner, the aggregator, and the analysis script.
- Added a reference comparison protocol. A design note documents how to build and
  run the SPAA/TBB baseline, the algorithm-id correspondence, and the
  correctness-before-performance rule, and a comparison script runs the project
  CLI and the baseline on the same input, normalizes both histograms, and reports
  MATCH or MISMATCH. The script's normalization, match, and error paths were
  exercised locally with a replayed baseline histogram.

### Notes

- CUDA kernels are guarded for non-CUDA development machines. Local validation
  checks host code, CLI behavior, documentation, and unavailable-backend paths;
  kernel compilation and GPU timing are expected on the H100 cluster build.
