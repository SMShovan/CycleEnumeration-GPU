#include "cycle_enum/core/graph.hpp"
#include "cycle_enum/core/graph_view.hpp"
#include "cycle_enum/core/histogram.hpp"
#include "cycle_enum/cuda/cuda_config.hpp"
#include "cycle_enum/cuda/cuda_work_queue.hpp"
#include "cycle_enum/sequential/bruteforce.hpp"

#include <cstddef>
#include <cstdint>
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

// s=0,a=1,b=2,w=3,c=4. Only <=4 cycle rooted at 0 is 0->3->4->0 (length 3); the
// cycle that would unblock vertex 3 has length 5. The certify backend undercounts
// this; the rank backend must get it exactly.
cycle_enum::GraphView counterexample_view() {
  return view_from_edges(
      {
          {0, 1, {0}}, {1, 2, {0}}, {2, 3, {0}},
          {3, 4, {0}}, {4, 0, {0}}, {0, 3, {0}},
      },
      5);
}

cycle_enum::GraphView clique_view(const std::size_t n) {
  std::vector<cycle_enum::TemporalEdge> edges;
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      if (i != j) {
        edges.push_back({static_cast<cycle_enum::ExternalVertexId>(i),
                         static_cast<cycle_enum::ExternalVertexId>(j),
                         {0}});
      }
    }
  }
  return view_from_edges(std::move(edges), n);
}

}  // namespace

// The headline property: with the budget off, the rank backend is exact for all
// cycles up to L, even on the counterexample where the certify backend undercounts.
TEST(CudaRankTest, CounterexampleIsExactUnderCap) {
  if (cuda_unavailable()) {
    GTEST_SKIP() << "no CUDA device visible";
  }
  const cycle_enum::GraphView view = counterexample_view();
  const std::size_t max_length = 4;

  cycle_enum::cuda::UncertaintyReport report;
  const cycle_enum::CycleHistogram rank =
      cycle_enum::cuda::count_simple_cycles_johnson_rank(
          view, 0, max_length, nullptr, /*step_budget=*/0, &report);
  const cycle_enum::CycleHistogram brute =
      cycle_enum::sequential::count_simple_cycles_bruteforce(view, max_length);

  EXPECT_EQ(rank, brute);
  EXPECT_EQ(rank.count(3), 1U);          // 0->3->4->0, the cycle certify misses
  EXPECT_EQ(report.flagged_count, 0U);   // exact: nothing flagged with budget off
}

// Exact equality with brute force and the non-blocking kernel on a dense graph,
// across several caps, with zero flagged roots.
TEST(CudaRankTest, ExactMatchesBruteForceOnDenseGraph) {
  if (cuda_unavailable()) {
    GTEST_SKIP() << "no CUDA device visible";
  }
  const cycle_enum::GraphView view = clique_view(6);
  for (const std::size_t max_length : {3U, 4U, 5U}) {
    cycle_enum::cuda::UncertaintyReport report;
    const cycle_enum::CycleHistogram rank =
        cycle_enum::cuda::count_simple_cycles_johnson_rank(
            view, 0, max_length, nullptr, 0, &report);
    const cycle_enum::CycleHistogram non_blocking =
        cycle_enum::cuda::count_simple_cycles_johnson_work_queue(
            view, 0, max_length, nullptr, false, nullptr);
    const cycle_enum::CycleHistogram brute =
        cycle_enum::sequential::count_simple_cycles_bruteforce(view, max_length);

    EXPECT_EQ(non_blocking, brute) << "max_length=" << max_length;
    EXPECT_EQ(rank, brute) << "max_length=" << max_length;
    EXPECT_EQ(report.flagged_count, 0U) << "max_length=" << max_length;
  }
}

// With the budget on, the result is a lower bound and flags truncated roots; a
// large budget reproduces the exact result with nothing flagged.
TEST(CudaRankTest, StepBudgetIsLowerBoundAndFlags) {
  if (cuda_unavailable()) {
    GTEST_SKIP() << "no CUDA device visible";
  }
  const cycle_enum::GraphView view = clique_view(6);
  const std::size_t max_length = 5;

  const cycle_enum::CycleHistogram exact =
      cycle_enum::cuda::count_simple_cycles_johnson_rank(view, 0, max_length,
                                                         nullptr, 0, nullptr);

  cycle_enum::cuda::UncertaintyReport tight;
  const cycle_enum::CycleHistogram budgeted =
      cycle_enum::cuda::count_simple_cycles_johnson_rank(
          view, 0, max_length, nullptr, /*step_budget=*/5, &tight);
  for (const auto& [length, count] : exact.entries()) {
    EXPECT_LE(budgeted.count(length), count) << "length=" << length;
  }
  EXPECT_GE(tight.flagged_count, 1U);

  cycle_enum::cuda::UncertaintyReport loose;
  const cycle_enum::CycleHistogram huge =
      cycle_enum::cuda::count_simple_cycles_johnson_rank(
          view, 0, max_length, nullptr,
          /*step_budget=*/1000000000ULL, &loose);
  EXPECT_EQ(huge, exact);
  EXPECT_EQ(loose.flagged_count, 0U);
}
