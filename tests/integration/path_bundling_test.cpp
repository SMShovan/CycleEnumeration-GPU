#include "cycle_enum/core/graph.hpp"
#include "cycle_enum/core/graph_view.hpp"
#include "cycle_enum/sequential/bruteforce.hpp"
#include "cycle_enum/sequential/temporal_johnson.hpp"

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

TEST(PathBundlingTest, MatchesOracleWithManyAssignmentsOnOneVertexPath) {
  const cycle_enum::GraphView view = view_from_edges(
      {
          {0, 1, {1}},
          {1, 2, {2, 3, 4}},
          {2, 0, {5, 6}},
      },
      3);

  const cycle_enum::sequential::TemporalJohnsonResult result =
      cycle_enum::sequential::count_temporal_cycles_johnson_with_stats(view,
                                                                       10);

  EXPECT_EQ(result.histogram,
            cycle_enum::sequential::count_temporal_cycles_bruteforce(view, 10));
  EXPECT_EQ(result.histogram.count(3), 6U);
  EXPECT_GT(result.stats.bundled_arrival_timestamps,
            result.stats.dfs_states);
}

TEST(PathBundlingTest, PreservesDuplicateTimestampMultiplicity) {
  const cycle_enum::GraphView view = view_from_edges(
      {
          {0, 1, {1, 1}},
          {1, 2, {2, 2}},
          {2, 0, {3}},
      },
      3);

  const cycle_enum::CycleHistogram histogram =
      cycle_enum::sequential::count_temporal_cycles_johnson(view, 10);

  EXPECT_EQ(histogram,
            cycle_enum::sequential::count_temporal_cycles_bruteforce(view, 10));
  EXPECT_EQ(histogram.count(3), 4U);
  EXPECT_EQ(histogram.total(), 4U);
}

}  // namespace
