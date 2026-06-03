#include "cycle_enum/core/graph.hpp"
#include "cycle_enum/core/graph_view.hpp"
#include "cycle_enum/engine/histogram_engine.hpp"
#include "cycle_enum/sequential/johnson.hpp"

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

TEST(HistogramEngineTest, CountMatchesSequentialRecompute) {
  const cycle_enum::GraphView view = view_from_edges(
      {
          {0, 1, {0}}, {1, 2, {0}}, {2, 0, {0}}, {1, 0, {0}},
      },
      3);

  EXPECT_EQ(cycle_enum::engine::count_histogram(view),
            cycle_enum::sequential::count_simple_cycles_johnson(view));
  EXPECT_EQ(cycle_enum::engine::count_histogram(view, std::size_t{2}),
            cycle_enum::sequential::count_simple_cycles_johnson(view,
                                                                std::size_t{2}));
}

}  // namespace
