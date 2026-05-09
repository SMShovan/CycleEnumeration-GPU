#include "cycle_enum/core/graph.hpp"
#include "cycle_enum/core/graph_view.hpp"
#include "cycle_enum/openmp/openmp_config.hpp"
#include "cycle_enum/openmp/openmp_temporal_read_tarjan.hpp"
#include "cycle_enum/sequential/temporal_read_tarjan.hpp"

#include <stdexcept>

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

cycle_enum::GraphView representative_temporal_view() {
  return view_from_edges(
      {
          {0, 1, {1, 1}},
          {1, 0, {4}},
          {1, 2, {2, 3}},
          {2, 0, {4, 5}},
          {2, 3, {4}},
          {3, 1, {6}},
      },
      4);
}

TEST(OpenMPTemporalReadTarjanTest,
     SingleThreadMatchesSequentialTemporalReadTarjan) {
  const cycle_enum::GraphView view = representative_temporal_view();

  EXPECT_EQ(cycle_enum::openmp::count_temporal_cycles_read_tarjan(view, 10, 1),
            cycle_enum::sequential::count_temporal_cycles_read_tarjan(view,
                                                                      10));
  EXPECT_EQ(
      cycle_enum::openmp::count_temporal_cycles_read_tarjan(view, 10, 1, 2),
      cycle_enum::sequential::count_temporal_cycles_read_tarjan(view, 10, 2));
}

TEST(OpenMPTemporalReadTarjanTest, SingleThreadReportsStats) {
  const cycle_enum::GraphView view = representative_temporal_view();

  const cycle_enum::openmp::TemporalReadTarjanResult result =
      cycle_enum::openmp::count_temporal_cycles_read_tarjan_with_stats(view, 10,
                                                                       1);

  EXPECT_EQ(result.histogram,
            cycle_enum::sequential::count_temporal_cycles_read_tarjan(view,
                                                                      10));
  EXPECT_GT(result.stats.dfs_states, 0U);
  EXPECT_GT(result.stats.timestamp_extensions, 0U);
}

TEST(OpenMPTemporalReadTarjanTest, MultiThreadRequestRequiresOpenMPRuntime) {
  const cycle_enum::GraphView view = representative_temporal_view();

  if (!cycle_enum::openmp::available()) {
    EXPECT_THROW((void)cycle_enum::openmp::count_temporal_cycles_read_tarjan(
                     view, 10, 2),
                 std::runtime_error);
    GTEST_SKIP() << "OpenMP runtime is not available in this build";
  }

  EXPECT_EQ(cycle_enum::openmp::count_temporal_cycles_read_tarjan(view, 10, 2),
            cycle_enum::sequential::count_temporal_cycles_read_tarjan(view,
                                                                      10));
}

TEST(OpenMPTemporalReadTarjanTest, RejectsInvalidConfiguration) {
  const cycle_enum::GraphView view = representative_temporal_view();

  EXPECT_THROW(
      (void)cycle_enum::openmp::count_temporal_cycles_read_tarjan(view, 10, 0),
      std::invalid_argument);
  EXPECT_THROW(
      (void)cycle_enum::openmp::count_temporal_cycles_read_tarjan(view, -1, 1),
      std::invalid_argument);
  EXPECT_THROW(
      (void)cycle_enum::openmp::count_temporal_cycles_read_tarjan(view, 10, 1,
                                                                  1),
      std::invalid_argument);
}

}  // namespace
