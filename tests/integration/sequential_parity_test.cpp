#include "cycle_enum/core/graph.hpp"
#include "cycle_enum/core/graph_view.hpp"
#include "cycle_enum/sequential/bruteforce.hpp"
#include "cycle_enum/sequential/johnson.hpp"
#include "cycle_enum/sequential/read_tarjan.hpp"
#include "cycle_enum/sequential/temporal_johnson.hpp"
#include "cycle_enum/sequential/temporal_read_tarjan.hpp"

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

TEST(SequentialParityTest, StaticAlgorithmsAgreeOnRepresentativeGraphs) {
  const std::vector<cycle_enum::GraphView> views = {
      view_from_edges(
          {
              {0, 1, {0}},
              {1, 0, {0}},
              {1, 2, {0}},
              {2, 0, {0}},
              {2, 3, {0}},
              {3, 1, {0}},
          },
          4),
      view_from_edges(
          {
              {0, 1, {0}},
              {1, 2, {0}},
              {2, 3, {0}},
          },
          4),
  };

  for (const cycle_enum::GraphView& view : views) {
    const cycle_enum::CycleHistogram oracle =
        cycle_enum::sequential::count_simple_cycles_bruteforce(view);
    EXPECT_EQ(cycle_enum::sequential::count_simple_cycles_johnson(view),
              oracle);
    EXPECT_EQ(cycle_enum::sequential::count_simple_cycles_read_tarjan(view),
              oracle);
  }
}

TEST(SequentialParityTest, SimpleTimeWindowAlgorithmsAgree) {
  const cycle_enum::GraphView view = view_from_edges(
      {
          {0, 1, {1}},
          {1, 2, {2, 3}},
          {2, 0, {4}},
          {1, 0, {5}},
      },
      3);

  const cycle_enum::CycleHistogram oracle =
      cycle_enum::sequential::count_time_window_cycles_bruteforce(view, 4);

  EXPECT_EQ(cycle_enum::sequential::count_time_window_cycles_johnson(view, 4),
            oracle);
  EXPECT_EQ(
      cycle_enum::sequential::count_time_window_cycles_read_tarjan(view, 4),
      oracle);
  EXPECT_EQ(
      cycle_enum::sequential::count_time_window_cycles_read_tarjan(view, 4, 2),
      cycle_enum::sequential::count_time_window_cycles_bruteforce(view, 4, 2));
}

TEST(SequentialParityTest, TemporalAlgorithmsAgree) {
  const cycle_enum::GraphView view = view_from_edges(
      {
          {0, 1, {1, 1}},
          {1, 2, {2, 3}},
          {2, 0, {4, 5}},
          {1, 0, {6}},
      },
      3);

  const cycle_enum::CycleHistogram oracle =
      cycle_enum::sequential::count_temporal_cycles_bruteforce(view, 10);

  EXPECT_EQ(cycle_enum::sequential::count_temporal_cycles_johnson(view, 10),
            oracle);
  EXPECT_EQ(
      cycle_enum::sequential::count_temporal_cycles_read_tarjan(view, 10),
      oracle);
  EXPECT_EQ(
      cycle_enum::sequential::count_temporal_cycles_read_tarjan(view, 10, 2),
      cycle_enum::sequential::count_temporal_cycles_bruteforce(view, 10, 2));
}

}  // namespace
