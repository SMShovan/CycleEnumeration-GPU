#include "cycle_enum/core/graph.hpp"
#include "cycle_enum/core/graph_view.hpp"
#include "cycle_enum/sequential/bruteforce.hpp"
#include "cycle_enum/sequential/johnson.hpp"

#include <cstddef>
#include <random>
#include <stdexcept>
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

void expect_johnson_matches_oracle(const cycle_enum::GraphView& view) {
  EXPECT_EQ(cycle_enum::sequential::count_simple_cycles_johnson(view),
            cycle_enum::sequential::count_simple_cycles_bruteforce(view));
}

TEST(JohnsonStaticTest, MatchesOracleOnAcyclicGraph) {
  const cycle_enum::GraphView view = view_from_edges(
      {
          {0, 1, {0}},
          {1, 2, {0}},
          {0, 2, {0}},
      },
      3);

  expect_johnson_matches_oracle(view);
  EXPECT_EQ(cycle_enum::sequential::count_simple_cycles_johnson(view).total(),
            0U);
}

TEST(JohnsonStaticTest, MatchesOracleOnSingleTriangle) {
  const cycle_enum::GraphView view = view_from_edges(
      {
          {0, 1, {0}},
          {1, 2, {0}},
          {2, 0, {0}},
      },
      3);

  expect_johnson_matches_oracle(view);
}

TEST(JohnsonStaticTest, MatchesOracleOnOverlappingCycles) {
  const cycle_enum::GraphView view = view_from_edges(
      {
          {0, 1, {0}},
          {1, 0, {0}},
          {1, 2, {0}},
          {2, 0, {0}},
          {2, 3, {0}},
          {3, 1, {0}},
      },
      4);

  const cycle_enum::CycleHistogram histogram =
      cycle_enum::sequential::count_simple_cycles_johnson(view);

  expect_johnson_matches_oracle(view);
  EXPECT_EQ(histogram.count(2), 1U);
  EXPECT_EQ(histogram.count(3), 2U);
  EXPECT_EQ(histogram.count(4), 0U);
  EXPECT_EQ(histogram.total(), 3U);
}

TEST(JohnsonStaticTest, HonorsMaximumCycleLength) {
  const cycle_enum::GraphView view = view_from_edges(
      {
          {0, 1, {0}},
          {1, 0, {0}},
          {1, 2, {0}},
          {2, 0, {0}},
      },
      3);

  EXPECT_EQ(cycle_enum::sequential::count_simple_cycles_johnson(view, 2),
            cycle_enum::sequential::count_simple_cycles_bruteforce(view, 2));
}

TEST(JohnsonStaticTest, RejectsInvalidMaximumCycleLength) {
  const cycle_enum::GraphView view = view_from_edges({}, 0);

  EXPECT_THROW((void)cycle_enum::sequential::count_simple_cycles_johnson(view, 1),
               std::invalid_argument);
}

// Regression: bounded Johnson must agree with the oracle on graphs dense enough
// to contain cycles both shorter and longer than the cap. A naive depth cutoff
// combined with Johnson's blocked-list mechanism silently drops some shorter
// cycles; this guards against that.
TEST(JohnsonStaticTest, BoundedMatchesOracleOnRandomGraphs) {
  std::mt19937_64 rng(2024);
  std::uniform_real_distribution<double> coin(0.0, 1.0);

  for (int trial = 0; trial < 30; ++trial) {
    constexpr std::size_t kVertices = 8;
    std::vector<cycle_enum::TemporalEdge> edges;
    for (std::size_t u = 0; u < kVertices; ++u) {
      for (std::size_t v = 0; v < kVertices; ++v) {
        if (u != v && coin(rng) < 0.4) {
          edges.push_back({static_cast<cycle_enum::VertexId>(u),
                           static_cast<cycle_enum::VertexId>(v), {0}});
        }
      }
    }
    const cycle_enum::GraphView view = view_from_edges(std::move(edges),
                                                       kVertices);

    for (std::size_t max_length = 2; max_length <= 7; ++max_length) {
      EXPECT_EQ(
          cycle_enum::sequential::count_simple_cycles_johnson(view, max_length),
          cycle_enum::sequential::count_simple_cycles_bruteforce(view,
                                                                 max_length))
          << "trial " << trial << " max_length " << max_length;
    }
  }
}

}  // namespace
