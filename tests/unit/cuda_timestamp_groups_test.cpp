#include "cycle_enum/core/graph.hpp"
#include "cycle_enum/core/graph_view.hpp"
#include "cycle_enum/cuda/cuda_timestamp_groups.hpp"

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

TEST(CudaTimestampGroupsTest, CollapsesDuplicateTimestamps) {
  // Edge 0->1 carries two events at t=5 and one at t=9.
  const cycle_enum::GraphView view = view_from_edges(
      {
          {0, 1, {5, 5, 9}},
          {1, 0, {7}},
      },
      2);

  const cycle_enum::cuda::TimestampGroups groups =
      cycle_enum::cuda::build_outgoing_timestamp_groups(view);

  ASSERT_EQ(groups.edge_group_offsets.size(),
            view.outgoing_edges().size() + 1);
  EXPECT_EQ(groups.edge_group_offsets.front(), 0u);
  EXPECT_EQ(groups.edge_group_offsets.back(), groups.group_values.size());
  EXPECT_EQ(groups.group_values.size(), groups.group_counts.size());

  // Every edge's groups must be strictly increasing and its counts must sum to
  // that edge's original timestamp count.
  const std::vector<cycle_enum::AdjacencyEntry>& edges = view.outgoing_edges();
  for (std::size_t slot = 0; slot < edges.size(); ++slot) {
    const cycle_enum::cuda::DeviceOffset begin = groups.edge_group_offsets[slot];
    const cycle_enum::cuda::DeviceOffset end =
        groups.edge_group_offsets[slot + 1];
    cycle_enum::cuda::DeviceOffset total = 0;
    for (cycle_enum::cuda::DeviceOffset g = begin; g < end; ++g) {
      total += groups.group_counts[g];
      if (g + 1 < end) {
        EXPECT_LT(groups.group_values[g], groups.group_values[g + 1]);
      }
    }
    const std::size_t original =
        edges[slot].timestamp_end - edges[slot].timestamp_begin;
    EXPECT_EQ(total, original);
  }

  // Edge 0->1 (slot 0) collapses {5,5,9} into two groups: (5, x2) and (9, x1).
  ASSERT_GE(groups.group_values.size(), 2u);
  EXPECT_EQ(groups.group_values[0], 5);
  EXPECT_EQ(groups.group_counts[0], 2u);
  EXPECT_EQ(groups.group_values[1], 9);
  EXPECT_EQ(groups.group_counts[1], 1u);
}

TEST(CudaTimestampGroupsTest, HandlesGraphWithoutEdges) {
  const cycle_enum::GraphView view =
      cycle_enum::build_graph_view(cycle_enum::TemporalGraph({}, {}));
  const cycle_enum::cuda::TimestampGroups groups =
      cycle_enum::cuda::build_outgoing_timestamp_groups(view);
  EXPECT_EQ(groups.edge_group_offsets, (std::vector<cycle_enum::cuda::DeviceOffset>{0}));
  EXPECT_TRUE(groups.group_values.empty());
  EXPECT_TRUE(groups.group_counts.empty());
}

}  // namespace
