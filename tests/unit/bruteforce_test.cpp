#include "cycle_enum/core/graph.hpp"
#include "cycle_enum/core/graph_view.hpp"
#include "cycle_enum/sequential/bruteforce.hpp"

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

TEST(BruteForceTest, CountsDirectedTriangleOnce) {
  const cycle_enum::GraphView view = view_from_edges(
      {
          {0, 1, {0}},
          {1, 2, {0}},
          {2, 0, {0}},
      },
      3);

  const cycle_enum::CycleHistogram histogram =
      cycle_enum::sequential::count_simple_cycles_bruteforce(view);

  EXPECT_EQ(histogram.count(3), 1U);
  EXPECT_EQ(histogram.total(), 1U);
}

TEST(BruteForceTest, CountsTwoVertexCycle) {
  const cycle_enum::GraphView view = view_from_edges(
      {
          {0, 1, {0}},
          {1, 0, {0}},
      },
      2);

  const cycle_enum::CycleHistogram histogram =
      cycle_enum::sequential::count_simple_cycles_bruteforce(view);

  EXPECT_EQ(histogram.count(2), 1U);
  EXPECT_EQ(histogram.total(), 1U);
}

TEST(BruteForceTest, CountsMultipleDirectedCyclesByLength) {
  const cycle_enum::GraphView view = view_from_edges(
      {
          {0, 1, {0}},
          {1, 0, {0}},
          {1, 2, {0}},
          {2, 0, {0}},
      },
      3);

  const cycle_enum::CycleHistogram histogram =
      cycle_enum::sequential::count_simple_cycles_bruteforce(view);

  EXPECT_EQ(histogram.count(2), 1U);
  EXPECT_EQ(histogram.count(3), 1U);
  EXPECT_EQ(histogram.total(), 2U);
}

TEST(BruteForceTest, HonorsMaximumCycleLength) {
  const cycle_enum::GraphView view = view_from_edges(
      {
          {0, 1, {0}},
          {1, 0, {0}},
          {1, 2, {0}},
          {2, 0, {0}},
      },
      3);

  const cycle_enum::CycleHistogram histogram =
      cycle_enum::sequential::count_simple_cycles_bruteforce(view, 2);

  EXPECT_EQ(histogram.count(2), 1U);
  EXPECT_EQ(histogram.count(3), 0U);
  EXPECT_EQ(histogram.total(), 1U);
}

TEST(BruteForceTest, RejectsInvalidMaximumCycleLength) {
  const cycle_enum::GraphView view = view_from_edges({}, 0);

  EXPECT_THROW(
      (void)cycle_enum::sequential::count_simple_cycles_bruteforce(view, 1),
      std::invalid_argument);
}

}  // namespace

