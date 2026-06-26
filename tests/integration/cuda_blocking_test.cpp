#include "cycle_enum/core/graph.hpp"
#include "cycle_enum/core/graph_view.hpp"
#include "cycle_enum/core/histogram.hpp"
#include "cycle_enum/cuda/cuda_config.hpp"
#include "cycle_enum/cuda/cuda_work_queue.hpp"
#include "cycle_enum/sequential/bruteforce.hpp"

#include <cstddef>
#include <vector>

#include <gtest/gtest.h>

namespace {

cycle_enum::GraphView view_from_edges(
    std::vector<cycle_enum::TemporalEdge> edges,
    const std::size_t vertex_count) {
  std::vector<cycle_enum::ExternalVertexId> external_ids;
  external_ids.reserve(vertex_count);
  for (std::size_t vertex = 0; vertex < vertex_count; ++vertex) {
    external_ids.push_back(static_cast<cycle_enum::ExternalVertexId>(vertex));
  }
  return cycle_enum::build_graph_view(
      cycle_enum::TemporalGraph(std::move(external_ids), std::move(edges)));
}

[[nodiscard]] bool cuda_unavailable() {
  return !cycle_enum::cuda::compiled_with_cuda() ||
         cycle_enum::cuda::device_count() == 0;
}

// Counterexample structure: s=0, a=1, b=2, w=3, c=4. Edges 0->1->2->3->4->0
// plus the shortcut 0->3. The only cycle of length <= 4 rooted at 0 is
// 0->3->4->0 (length 3). With a cap of 4, Johnson blocking marks vertex 3 while
// exploring 0->1->2->3 (depth 4) and never unblocks it, because the cycle that
// would unblock it (0->1->2->3->4->0) has length 5, so the length-3 cycle is
// missed and root 0 must be flagged.
cycle_enum::GraphView counterexample_view() {
  return view_from_edges(
      {
          {0, 1, {0}}, {1, 2, {0}}, {2, 3, {0}},
          {3, 4, {0}}, {4, 0, {0}}, {0, 3, {0}},
      },
      5);
}

// A few overlapping cycles among vertices 0..4 (longest simple cycle <= 5).
cycle_enum::GraphView dense_small_view() {
  return view_from_edges(
      {
          {0, 1, {0}}, {1, 2, {0}}, {2, 0, {0}}, {1, 3, {0}},
          {3, 0, {0}}, {2, 4, {0}}, {4, 1, {0}}, {3, 4, {0}},
      },
      5);
}

}  // namespace

// When the length cap cannot bite (L exceeds the longest simple cycle), the
// blocking backend must equal the non-blocking backend and the brute-force
// oracle, with zero flagged roots (a certified-exact run).
TEST(CudaBlockingTest, NoCapMatchesNonBlockingAndBruteForce) {
  if (cuda_unavailable()) {
    GTEST_SKIP() << "no CUDA device visible";
  }
  const cycle_enum::GraphView view = dense_small_view();
  const std::size_t max_length = 8;

  cycle_enum::cuda::UncertaintyReport report;
  const cycle_enum::CycleHistogram blocking =
      cycle_enum::cuda::count_simple_cycles_johnson_work_queue(
          view, 0, max_length, nullptr, /*blocking=*/true, &report);
  const cycle_enum::CycleHistogram non_blocking =
      cycle_enum::cuda::count_simple_cycles_johnson_work_queue(
          view, 0, max_length, nullptr, /*blocking=*/false, nullptr);
  const cycle_enum::CycleHistogram brute =
      cycle_enum::sequential::count_simple_cycles_bruteforce(view, max_length);

  EXPECT_EQ(non_blocking, brute);
  EXPECT_EQ(blocking, non_blocking);
  EXPECT_EQ(report.flagged_count, 0U);
  EXPECT_EQ(report.total_roots, view.vertex_count());
}

// Under a tight cap, blocking is a strict lower bound and flags the affected
// root, while the non-blocking backend stays exact.
TEST(CudaBlockingTest, CounterexampleUndercountsAndFlagsRoot) {
  if (cuda_unavailable()) {
    GTEST_SKIP() << "no CUDA device visible";
  }
  const cycle_enum::GraphView view = counterexample_view();
  const std::size_t max_length = 4;

  cycle_enum::cuda::UncertaintyReport report;
  const cycle_enum::CycleHistogram blocking =
      cycle_enum::cuda::count_simple_cycles_johnson_work_queue(
          view, 0, max_length, nullptr, /*blocking=*/true, &report);
  const cycle_enum::CycleHistogram non_blocking =
      cycle_enum::cuda::count_simple_cycles_johnson_work_queue(
          view, 0, max_length, nullptr, /*blocking=*/false, nullptr);
  const cycle_enum::CycleHistogram brute =
      cycle_enum::sequential::count_simple_cycles_bruteforce(view, max_length);

  // Non-blocking is exact for cycles up to the cap.
  EXPECT_EQ(non_blocking, brute);
  EXPECT_EQ(non_blocking.count(3), 1U);  // 0->3->4->0

  // Blocking misses at least that cycle and flags the root, but never overcounts.
  EXPECT_NE(blocking, non_blocking);
  EXPECT_LT(blocking.total(), non_blocking.total());
  EXPECT_GE(report.flagged_count, 1U);
  for (const auto& [length, count] : non_blocking.entries()) {
    EXPECT_LE(blocking.count(length), count);
  }
}
