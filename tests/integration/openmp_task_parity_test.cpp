#include "cycle_enum/core/graph.hpp"
#include "cycle_enum/core/graph_view.hpp"
#include "cycle_enum/openmp/openmp_config.hpp"
#include "cycle_enum/openmp/openmp_read_tarjan.hpp"
#include "cycle_enum/openmp/openmp_read_tarjan_tasks.hpp"
#include "cycle_enum/sequential/read_tarjan.hpp"

#include <stdexcept>
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

// A denser fixture so several cutoff depths exercise real task spawning.
cycle_enum::GraphView representative_view() {
  return view_from_edges(
      {
          {0, 1, {0}},
          {1, 0, {0}},
          {1, 2, {0}},
          {2, 0, {0}},
          {2, 3, {0}},
          {3, 1, {0}},
          {0, 3, {0}},
          {3, 0, {0}},
          {2, 1, {0}},
      },
      4);
}

TEST(OpenMPTaskParityTest, SingleThreadMatchesSequentialReadTarjan) {
  const cycle_enum::GraphView view = representative_view();
  const cycle_enum::CycleHistogram expected =
      cycle_enum::sequential::count_simple_cycles_read_tarjan(view);

  for (const std::size_t cutoff : {std::size_t{1}, std::size_t{2},
                                   std::size_t{3}, std::size_t{4}}) {
    EXPECT_EQ(cycle_enum::openmp::count_simple_cycles_read_tarjan_tasks(
                  view, 1, std::nullopt, cutoff),
              expected)
        << "cutoff=" << cutoff;
  }
}

TEST(OpenMPTaskParityTest, RespectsMaxCycleLength) {
  const cycle_enum::GraphView view = representative_view();

  EXPECT_EQ(cycle_enum::openmp::count_simple_cycles_read_tarjan_tasks(
                view, 1, std::size_t{3}, std::size_t{2}),
            cycle_enum::sequential::count_simple_cycles_read_tarjan(
                view, std::size_t{3}));
}

TEST(OpenMPTaskParityTest, MultiThreadMatchesCoarseGrained) {
  const cycle_enum::GraphView view = representative_view();
  const cycle_enum::CycleHistogram expected =
      cycle_enum::sequential::count_simple_cycles_read_tarjan(view);

  if (!cycle_enum::openmp::available()) {
    EXPECT_THROW(
        (void)cycle_enum::openmp::count_simple_cycles_read_tarjan_tasks(view, 2),
        std::runtime_error);
    GTEST_SKIP() << "OpenMP runtime is not available in this build";
  }

  for (const int threads : {2, 4}) {
    for (const std::size_t cutoff : {std::size_t{1}, std::size_t{2},
                                     std::size_t{3}}) {
      EXPECT_EQ(cycle_enum::openmp::count_simple_cycles_read_tarjan_tasks(
                    view, threads, std::nullopt, cutoff),
                expected)
          << "threads=" << threads << " cutoff=" << cutoff;
      EXPECT_EQ(cycle_enum::openmp::count_simple_cycles_read_tarjan_tasks(
                    view, threads, std::nullopt, cutoff),
                cycle_enum::openmp::count_simple_cycles_read_tarjan(
                    view, threads));
    }
  }
}

TEST(OpenMPTaskParityTest, RejectsInvalidConfiguration) {
  const cycle_enum::GraphView view = representative_view();

  EXPECT_THROW(
      (void)cycle_enum::openmp::count_simple_cycles_read_tarjan_tasks(view, 0),
      std::invalid_argument);
  EXPECT_THROW(
      (void)cycle_enum::openmp::count_simple_cycles_read_tarjan_tasks(
          view, 1, std::size_t{1}),
      std::invalid_argument);
  EXPECT_THROW(
      (void)cycle_enum::openmp::count_simple_cycles_read_tarjan_tasks(
          view, 1, std::nullopt, std::size_t{0}),
      std::invalid_argument);
}

}  // namespace
