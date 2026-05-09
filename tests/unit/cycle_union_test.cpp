#include "cycle_enum/core/graph.hpp"
#include "cycle_enum/core/graph_view.hpp"
#include "cycle_enum/sequential/cycle_union.hpp"

#include <limits>
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

TEST(CycleUnionTest, KeepsCycleVerticesAndDropsDeadBranch) {
  const cycle_enum::GraphView view = view_from_edges(
      {
          {0, 1, {1}},
          {1, 2, {2}},
          {2, 0, {3}},
          {1, 3, {2}},
      },
      4);

  const cycle_enum::sequential::CycleUnion cycle_union =
      cycle_enum::sequential::compute_temporal_cycle_union(
          view, cycle_enum::sequential::CycleUnionRequest{0, 1, 1, 10});

  EXPECT_TRUE(cycle_union.contains(0));
  EXPECT_TRUE(cycle_union.contains(1));
  EXPECT_TRUE(cycle_union.contains(2));
  EXPECT_FALSE(cycle_union.contains(3));
  EXPECT_EQ(cycle_union.included_count(), 3U);
}

TEST(CycleUnionTest, SupportsTwoVertexCycle) {
  const cycle_enum::GraphView view = view_from_edges(
      {
          {0, 1, {1}},
          {1, 0, {2}},
      },
      2);

  const cycle_enum::sequential::CycleUnion cycle_union =
      cycle_enum::sequential::compute_temporal_cycle_union(
          view, cycle_enum::sequential::CycleUnionRequest{0, 1, 1, 10});

  EXPECT_EQ(cycle_union.included_vertices(),
            (std::vector<cycle_enum::VertexId>{0, 1}));
}

TEST(CycleUnionTest, ReturnsEmptyWhenRootCannotBeReached) {
  const cycle_enum::GraphView view = view_from_edges(
      {
          {0, 1, {1}},
          {1, 2, {2}},
      },
      3);

  const cycle_enum::sequential::CycleUnion cycle_union =
      cycle_enum::sequential::compute_temporal_cycle_union(
          view, cycle_enum::sequential::CycleUnionRequest{0, 1, 1, 10});

  EXPECT_EQ(cycle_union.included_count(), 0U);
  EXPECT_TRUE(cycle_union.included_vertices().empty());
}

TEST(CycleUnionTest, HonorsTimeWindow) {
  const cycle_enum::GraphView view = view_from_edges(
      {
          {0, 1, {1}},
          {1, 2, {2}},
          {2, 0, {7}},
      },
      3);

  EXPECT_EQ(cycle_enum::sequential::compute_temporal_cycle_union(
                view, cycle_enum::sequential::CycleUnionRequest{0, 1, 1, 5})
                .included_count(),
            0U);
  EXPECT_EQ(cycle_enum::sequential::compute_temporal_cycle_union(
                view, cycle_enum::sequential::CycleUnionRequest{0, 1, 1, 6})
                .included_count(),
            3U);
}

TEST(CycleUnionTest, RejectsInvalidRequest) {
  const cycle_enum::GraphView view = view_from_edges({}, 2);

  EXPECT_THROW((void)cycle_enum::sequential::compute_temporal_cycle_union(
                   view,
                   cycle_enum::sequential::CycleUnionRequest{2, 0, 0, 1}),
               std::out_of_range);
  EXPECT_THROW((void)cycle_enum::sequential::compute_temporal_cycle_union(
                   view,
                   cycle_enum::sequential::CycleUnionRequest{0, 2, 0, 1}),
               std::out_of_range);
  EXPECT_THROW((void)cycle_enum::sequential::compute_temporal_cycle_union(
                   view,
                   cycle_enum::sequential::CycleUnionRequest{0, 1, 0, -1}),
               std::invalid_argument);
  EXPECT_THROW(
      (void)cycle_enum::sequential::compute_temporal_cycle_union(
          view, cycle_enum::sequential::CycleUnionRequest{
                    0, 1, std::numeric_limits<cycle_enum::Timestamp>::max(),
                    1}),
      std::overflow_error);
}

}  // namespace
