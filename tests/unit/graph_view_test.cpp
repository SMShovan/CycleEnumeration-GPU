#include "cycle_enum/core/graph.hpp"
#include "cycle_enum/core/graph_view.hpp"

#include <stdexcept>

#include <gtest/gtest.h>

namespace {

cycle_enum::TemporalGraph sample_graph() {
  return cycle_enum::read_temporal_graph(
      std::filesystem::path(CYCLE_ENUM_TEST_DATA_DIR) / "sample_temporal.txt");
}

TEST(GraphViewTest, BuildsCsrAndCscOffsets) {
  const cycle_enum::TemporalGraph graph = sample_graph();
  const cycle_enum::GraphView view = cycle_enum::build_graph_view(graph);

  EXPECT_EQ(view.vertex_count(), 3U);
  EXPECT_EQ(view.edge_count(), 3U);
  EXPECT_EQ(view.timestamp_count(), 4U);

  EXPECT_EQ(view.outgoing_offsets(),
            (std::vector<std::size_t>{0, 1, 2, 3}));
  EXPECT_EQ(view.incoming_offsets(),
            (std::vector<std::size_t>{0, 2, 3, 3}));

  EXPECT_EQ(view.out_degree(0), 1U);
  EXPECT_EQ(view.in_degree(0), 2U);
  EXPECT_EQ(view.in_degree(2), 0U);
}

TEST(GraphViewTest, PreservesOutgoingAdjacencyAndFlatTimestamps) {
  const cycle_enum::GraphView view = cycle_enum::build_graph_view(sample_graph());

  ASSERT_EQ(view.outgoing_edges().size(), 3U);
  EXPECT_EQ(view.timestamps(),
            (std::vector<cycle_enum::Timestamp>{3, 5, 7, 6}));

  const cycle_enum::AdjacencyEntry& first = view.outgoing_edges()[0];
  EXPECT_EQ(first.vertex, 1U);
  EXPECT_EQ(first.edge_id, 0U);
  EXPECT_EQ(first.timestamp_begin, 0U);
  EXPECT_EQ(first.timestamp_end, 2U);

  const cycle_enum::AdjacencyEntry& second = view.outgoing_edges()[1];
  EXPECT_EQ(second.vertex, 0U);
  EXPECT_EQ(second.edge_id, 1U);
  EXPECT_EQ(second.timestamp_begin, 2U);
  EXPECT_EQ(second.timestamp_end, 3U);
}

TEST(GraphViewTest, PreservesIncomingAdjacency) {
  const cycle_enum::GraphView view = cycle_enum::build_graph_view(sample_graph());

  ASSERT_EQ(view.incoming_edges().size(), 3U);

  const cycle_enum::AdjacencyEntry& into_zero_from_twenty =
      view.incoming_edges()[0];
  EXPECT_EQ(into_zero_from_twenty.vertex, 1U);
  EXPECT_EQ(into_zero_from_twenty.edge_id, 1U);
  EXPECT_EQ(into_zero_from_twenty.timestamp_begin, 2U);
  EXPECT_EQ(into_zero_from_twenty.timestamp_end, 3U);

  const cycle_enum::AdjacencyEntry& into_zero_from_thirty =
      view.incoming_edges()[1];
  EXPECT_EQ(into_zero_from_thirty.vertex, 2U);
  EXPECT_EQ(into_zero_from_thirty.edge_id, 2U);
  EXPECT_EQ(into_zero_from_thirty.timestamp_begin, 3U);
  EXPECT_EQ(into_zero_from_thirty.timestamp_end, 4U);
}

TEST(GraphViewTest, RejectsInvalidDegreeQueries) {
  const cycle_enum::GraphView view = cycle_enum::build_graph_view(sample_graph());

  EXPECT_THROW((void)view.out_degree(99), std::out_of_range);
  EXPECT_THROW((void)view.in_degree(99), std::out_of_range);
}

}  // namespace

