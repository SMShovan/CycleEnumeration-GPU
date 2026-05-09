#include "cycle_enum/core/graph.hpp"
#include "cycle_enum/core/graph_view.hpp"
#include "cycle_enum/sequential/bruteforce.hpp"
#include "cycle_enum/sequential/temporal_johnson.hpp"
#include "cycle_enum/sequential/temporal_read_tarjan.hpp"

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

void expect_matches_temporal_references(
    const cycle_enum::GraphView& view,
    const cycle_enum::Timestamp window_width,
    const std::optional<std::size_t> max_length = std::nullopt) {
  const cycle_enum::CycleHistogram read_tarjan =
      cycle_enum::sequential::count_temporal_cycles_read_tarjan(
          view, window_width, max_length);

  EXPECT_EQ(read_tarjan,
            cycle_enum::sequential::count_temporal_cycles_bruteforce(
                view, window_width, max_length));
  EXPECT_EQ(read_tarjan,
            cycle_enum::sequential::count_temporal_cycles_johnson(
                view, window_width, max_length));
}

TEST(TemporalReadTarjanTest, MatchesReferencesOnIncreasingTriangle) {
  const cycle_enum::GraphView view = view_from_edges(
      {
          {0, 1, {1}},
          {1, 2, {2}},
          {2, 0, {3}},
      },
      3);

  expect_matches_temporal_references(view, 10);
  EXPECT_EQ(cycle_enum::sequential::count_temporal_cycles_read_tarjan(view, 10)
                .count(3),
            1U);
}

TEST(TemporalReadTarjanTest, MatchesReferencesWithTimestampMultiplicity) {
  const cycle_enum::GraphView view = view_from_edges(
      {
          {0, 1, {1, 1}},
          {1, 2, {2, 3}},
          {2, 0, {4, 5}},
      },
      3);

  const cycle_enum::CycleHistogram histogram =
      cycle_enum::sequential::count_temporal_cycles_read_tarjan(view, 10);

  expect_matches_temporal_references(view, 10);
  EXPECT_EQ(histogram.count(3), 8U);
  EXPECT_EQ(histogram.total(), 8U);
}

TEST(TemporalReadTarjanTest, MatchesReferencesOnMixedCycleLengths) {
  const cycle_enum::GraphView view = view_from_edges(
      {
          {0, 1, {1}},
          {1, 0, {2}},
          {1, 2, {2}},
          {2, 0, {3}},
      },
      3);

  const cycle_enum::CycleHistogram histogram =
      cycle_enum::sequential::count_temporal_cycles_read_tarjan(view, 10);

  expect_matches_temporal_references(view, 10);
  EXPECT_EQ(histogram.count(2), 1U);
  EXPECT_EQ(histogram.count(3), 1U);
  EXPECT_EQ(histogram.total(), 2U);
}

TEST(TemporalReadTarjanTest, HonorsTimeWindowAndMaximumLength) {
  const cycle_enum::GraphView view = view_from_edges(
      {
          {0, 1, {1}},
          {1, 0, {5}},
          {1, 2, {2}},
          {2, 0, {4}},
      },
      3);

  expect_matches_temporal_references(view, 3, 3);
  EXPECT_EQ(cycle_enum::sequential::count_temporal_cycles_read_tarjan(view, 3)
                .count(2),
            0U);
  EXPECT_EQ(
      cycle_enum::sequential::count_temporal_cycles_read_tarjan(view, 3, 2)
          .count(3),
      0U);
}

TEST(TemporalReadTarjanTest, RejectsInvalidConfiguration) {
  const cycle_enum::GraphView view = view_from_edges({}, 0);

  EXPECT_THROW(
      (void)cycle_enum::sequential::count_temporal_cycles_read_tarjan(view, -1),
      std::invalid_argument);
  EXPECT_THROW(
      (void)cycle_enum::sequential::count_temporal_cycles_read_tarjan(view, 1,
                                                                      1),
      std::invalid_argument);
}

}  // namespace
