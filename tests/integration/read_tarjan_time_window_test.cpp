#include "cycle_enum/core/graph.hpp"
#include "cycle_enum/core/graph_view.hpp"
#include "cycle_enum/sequential/bruteforce.hpp"
#include "cycle_enum/sequential/read_tarjan.hpp"

#include <filesystem>
#include <stdexcept>

#include <gtest/gtest.h>

namespace {

cycle_enum::GraphView read_view(const std::string& filename) {
  const cycle_enum::TemporalGraph graph = cycle_enum::read_temporal_graph(
      std::filesystem::path(CYCLE_ENUM_TEST_DATA_DIR) / filename);
  return cycle_enum::build_graph_view(graph);
}

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

TEST(ReadTarjanTimeWindowTest, ReproducesBaselineReadmeSample) {
  const cycle_enum::GraphView view = read_view("reference_sample.txt");

  const cycle_enum::CycleHistogram read_tarjan =
      cycle_enum::sequential::count_time_window_cycles_read_tarjan(view, 3600);
  const cycle_enum::CycleHistogram oracle =
      cycle_enum::sequential::count_time_window_cycles_bruteforce(view, 3600);

  EXPECT_EQ(read_tarjan, oracle);
  EXPECT_EQ(read_tarjan.count(2), 1U);
  EXPECT_EQ(read_tarjan.count(3), 1U);
  EXPECT_EQ(read_tarjan.count(4), 2U);
  EXPECT_EQ(read_tarjan.count(5), 1U);
  EXPECT_EQ(read_tarjan.total(), 5U);
}

TEST(ReadTarjanTimeWindowTest, NarrowWindowCanRemoveCycle) {
  const cycle_enum::GraphView view = view_from_edges(
      {
          {0, 1, {0}},
          {1, 0, {10}},
      },
      2);

  EXPECT_EQ(
      cycle_enum::sequential::count_time_window_cycles_read_tarjan(view, 5)
          .total(),
      0U);
  EXPECT_EQ(
      cycle_enum::sequential::count_time_window_cycles_read_tarjan(view, 10)
          .count(2),
      1U);
}

TEST(ReadTarjanTimeWindowTest, HonorsMaximumCycleLength) {
  const cycle_enum::GraphView view = read_view("reference_sample.txt");

  EXPECT_EQ(cycle_enum::sequential::count_time_window_cycles_read_tarjan(
                view, 3600, 3),
            cycle_enum::sequential::count_time_window_cycles_bruteforce(
                view, 3600, 3));
}

TEST(ReadTarjanTimeWindowTest, RejectsInvalidMaximumCycleLength) {
  const cycle_enum::GraphView view = view_from_edges({}, 0);

  EXPECT_THROW(
      (void)cycle_enum::sequential::count_time_window_cycles_read_tarjan(
          view, 1, 1),
      std::invalid_argument);
}

}  // namespace

