#include "cycle_enum/core/graph.hpp"
#include "cycle_enum/core/graph_view.hpp"
#include "cycle_enum/cuda/cuda_graph.hpp"

#include <gtest/gtest.h>

namespace {

cycle_enum::TemporalGraph sample_graph() {
  return cycle_enum::read_temporal_graph(
      std::filesystem::path(CYCLE_ENUM_TEST_DATA_DIR) / "sample_temporal.txt");
}

}  // namespace

TEST(CudaGraphTest, PacksGraphViewIntoFixedWidthArrays) {
  const cycle_enum::GraphView view = cycle_enum::build_graph_view(sample_graph());
  const cycle_enum::cuda::CudaGraphData packed =
      cycle_enum::cuda::pack_graph_for_cuda(view);

  EXPECT_EQ(packed.vertex_count, 3U);
  EXPECT_EQ(packed.edge_count, 3U);
  EXPECT_EQ(packed.timestamp_count, 4U);

  EXPECT_EQ(packed.outgoing_offsets,
            (std::vector<cycle_enum::cuda::DeviceOffset>{0, 1, 2, 3}));
  EXPECT_EQ(packed.incoming_offsets,
            (std::vector<cycle_enum::cuda::DeviceOffset>{0, 2, 3, 3}));

  ASSERT_EQ(packed.outgoing_edges.size(), 3U);
  EXPECT_EQ(packed.outgoing_edges[0].vertex, 1U);
  EXPECT_EQ(packed.outgoing_edges[0].edge_id, 0U);
  EXPECT_EQ(packed.outgoing_edges[0].timestamp_begin, 0U);
  EXPECT_EQ(packed.outgoing_edges[0].timestamp_end, 2U);

  EXPECT_EQ(packed.timestamps,
            (std::vector<cycle_enum::Timestamp>{3, 5, 7, 6}));
}

TEST(CudaGraphTest, PacksOutgoingAdjacencyAsStructureOfArrays) {
  const cycle_enum::GraphView view = cycle_enum::build_graph_view(sample_graph());
  const cycle_enum::cuda::CudaGraphData packed =
      cycle_enum::cuda::pack_graph_for_cuda(view);

  // The SoA hot-path arrays must mirror the AoS layout entry by entry.
  ASSERT_EQ(packed.outgoing_neighbors.size(), packed.outgoing_edges.size());
  ASSERT_EQ(packed.outgoing_timestamp_begin.size(), packed.outgoing_edges.size());
  ASSERT_EQ(packed.outgoing_timestamp_end.size(), packed.outgoing_edges.size());

  for (std::size_t index = 0; index < packed.outgoing_edges.size(); ++index) {
    EXPECT_EQ(packed.outgoing_neighbors[index],
              packed.outgoing_edges[index].vertex);
    EXPECT_EQ(packed.outgoing_timestamp_begin[index],
              packed.outgoing_edges[index].timestamp_begin);
    EXPECT_EQ(packed.outgoing_timestamp_end[index],
              packed.outgoing_edges[index].timestamp_end);
  }
}

TEST(CudaGraphTest, PacksEmptyGraph) {
  const cycle_enum::GraphView view =
      cycle_enum::build_graph_view(cycle_enum::TemporalGraph({},
                                                             {}));
  const cycle_enum::cuda::CudaGraphData packed =
      cycle_enum::cuda::pack_graph_for_cuda(view);

  EXPECT_EQ(packed.vertex_count, 0U);
  EXPECT_EQ(packed.edge_count, 0U);
  EXPECT_EQ(packed.timestamp_count, 0U);
  EXPECT_EQ(packed.outgoing_offsets,
            (std::vector<cycle_enum::cuda::DeviceOffset>{0}));
  EXPECT_EQ(packed.incoming_offsets,
            (std::vector<cycle_enum::cuda::DeviceOffset>{0}));
  EXPECT_TRUE(packed.outgoing_edges.empty());
  EXPECT_TRUE(packed.incoming_edges.empty());
  EXPECT_TRUE(packed.timestamps.empty());
}
