#include "cycle_enum/core/graph.hpp"
#include "cycle_enum/core/graph_view.hpp"
#include "cycle_enum/dynamic/directed_graph.hpp"
#include "cycle_enum/dynamic/edge_change.hpp"

#include <cstddef>
#include <vector>

#include <gtest/gtest.h>

namespace {

cycle_enum::dynamic::DirectedGraph build(
    std::vector<cycle_enum::TemporalEdge> edges,
    const std::size_t vertex_count) {
  std::vector<cycle_enum::ExternalVertexId> external_ids;
  for (std::size_t v = 0; v < vertex_count; ++v) {
    external_ids.push_back(static_cast<cycle_enum::ExternalVertexId>(v));
  }
  return cycle_enum::dynamic::build_directed_graph(cycle_enum::build_graph_view(
      cycle_enum::TemporalGraph(std::move(external_ids), std::move(edges))));
}

TEST(DirectedGraphTest, BuildAndHasEdge) {
  const cycle_enum::dynamic::DirectedGraph g =
      build({{0, 1, {0}}, {1, 2, {0}}, {2, 0, {0}}}, 3);
  EXPECT_EQ(g.vertex_count, 3u);
  EXPECT_TRUE(cycle_enum::dynamic::has_edge(g, 0, 1));
  EXPECT_TRUE(cycle_enum::dynamic::has_edge(g, 2, 0));
  EXPECT_FALSE(cycle_enum::dynamic::has_edge(g, 1, 0));
  EXPECT_FALSE(cycle_enum::dynamic::has_edge(g, 0, 2));
}

TEST(DirectedGraphTest, ApplyBatchAddsAndRemovesEdges) {
  const cycle_enum::dynamic::DirectedGraph g =
      build({{0, 1, {0}}, {1, 2, {0}}, {2, 0, {0}}}, 3);

  cycle_enum::dynamic::EdgeBatch batch;
  batch.deletions = {{1, 2}};
  batch.insertions = {{0, 2}};

  const cycle_enum::dynamic::DirectedGraph updated =
      cycle_enum::dynamic::apply_batch(g, batch);

  EXPECT_FALSE(cycle_enum::dynamic::has_edge(updated, 1, 2)); // deleted
  EXPECT_TRUE(cycle_enum::dynamic::has_edge(updated, 0, 2));  // inserted
  EXPECT_TRUE(cycle_enum::dynamic::has_edge(updated, 0, 1));  // untouched
  EXPECT_TRUE(cycle_enum::dynamic::has_edge(updated, 2, 0));  // untouched
}

}  // namespace
