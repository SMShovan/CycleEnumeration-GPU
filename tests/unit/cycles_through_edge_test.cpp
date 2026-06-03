#include "cycle_enum/core/graph.hpp"
#include "cycle_enum/core/graph_view.hpp"
#include "cycle_enum/dynamic/cycles_through_edge.hpp"
#include "cycle_enum/dynamic/directed_graph.hpp"

#include <cstddef>
#include <cstdint>
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

// 0->1->2->0 (triangle) and 1->0 (a 2-cycle 0<->1).
cycle_enum::dynamic::DirectedGraph fixture() {
  return build({{0, 1, {0}}, {1, 2, {0}}, {2, 0, {0}}, {1, 0, {0}}}, 3);
}

std::vector<std::uint64_t> count_through(
    const cycle_enum::dynamic::DirectedGraph& g,
    cycle_enum::VertexId u, cycle_enum::VertexId v, std::size_t owner_id,
    const cycle_enum::dynamic::ChangedEdgeIndex& index, std::size_t max_len) {
  std::vector<std::uint64_t> counts(max_len + 1, 0);
  cycle_enum::dynamic::count_cycles_through_edge(g, u, v, owner_id, index,
                                                 max_len, counts);
  return counts;
}

TEST(CyclesThroughEdgeTest, CountsBothCyclesThroughAnEdge) {
  const cycle_enum::dynamic::DirectedGraph g = fixture();
  const cycle_enum::dynamic::ChangedEdgeIndex none({});

  // Edge 0->1 is on the 2-cycle (0->1->0) and the 3-cycle (0->1->2->0).
  const std::vector<std::uint64_t> c = count_through(g, 0, 1, 0, none, 8);
  EXPECT_EQ(c[2], 1u);
  EXPECT_EQ(c[3], 1u);
}

TEST(CyclesThroughEdgeTest, CountsEdgeOnlyOnLongerCycle) {
  const cycle_enum::dynamic::DirectedGraph g = fixture();
  const cycle_enum::dynamic::ChangedEdgeIndex none({});

  // Edge 2->0 is only on the 3-cycle (2->0->1->2).
  const std::vector<std::uint64_t> c = count_through(g, 2, 0, 0, none, 8);
  EXPECT_EQ(c[2], 0u);
  EXPECT_EQ(c[3], 1u);
}

TEST(CyclesThroughEdgeTest, RespectsMaxLength) {
  const cycle_enum::dynamic::DirectedGraph g = fixture();
  const cycle_enum::dynamic::ChangedEdgeIndex none({});

  // Capping at length 2 keeps only the 2-cycle through 0->1.
  const std::vector<std::uint64_t> c = count_through(g, 0, 1, 0, none, 2);
  EXPECT_EQ(c[2], 1u);
}

TEST(CyclesThroughEdgeTest, OwnershipSkipsCyclesWithSmallerIdEdge) {
  const cycle_enum::dynamic::DirectedGraph g = fixture();

  // Phase changes: edge (1,0) has id 0, edge (0,1) has id 1.
  const cycle_enum::dynamic::ChangedEdgeIndex index({{0, 1}, {1, 0}});
  // build_directed_graph / normalize sort by (src,tgt): (0,1) -> id 0, (1,0) -> id 1.
  // Anchor on (1,0) as owner_id 1; the 2-cycle also uses (0,1) with id 0 < 1,
  // so it is owned by (0,1) and must be skipped here.
  const std::vector<std::uint64_t> c = count_through(g, 1, 0, 1, index, 8);
  EXPECT_EQ(c[2], 0u); // 2-cycle skipped (owned by smaller-id edge 0->1)
}

}  // namespace
