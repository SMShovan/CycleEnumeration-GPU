#include "cycle_enum/core/graph.hpp"
#include "cycle_enum/core/graph_view.hpp"
#include "cycle_enum/openmp/openmp_config.hpp"
#include "cycle_enum/openmp/openmp_johnson.hpp"
#include "cycle_enum/sequential/johnson.hpp"

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

cycle_enum::GraphView representative_view() {
  return view_from_edges(
      {
          {0, 1, {0}},
          {1, 0, {0}},
          {1, 2, {0}},
          {2, 0, {0}},
          {2, 3, {0}},
          {3, 1, {0}},
      },
      4);
}

TEST(OpenMPJohnsonTest, SingleThreadMatchesSequentialJohnson) {
  const cycle_enum::GraphView view = representative_view();

  EXPECT_EQ(cycle_enum::openmp::count_simple_cycles_johnson(view, 1),
            cycle_enum::sequential::count_simple_cycles_johnson(view));
  EXPECT_EQ(cycle_enum::openmp::count_simple_cycles_johnson(view, 1, 2),
            cycle_enum::sequential::count_simple_cycles_johnson(view, 2));
}

TEST(OpenMPJohnsonTest, MultiThreadRequestRequiresOpenMPRuntime) {
  const cycle_enum::GraphView view = representative_view();

  if (!cycle_enum::openmp::available()) {
    EXPECT_THROW((void)cycle_enum::openmp::count_simple_cycles_johnson(view, 2),
                 std::runtime_error);
    GTEST_SKIP() << "OpenMP runtime is not available in this build";
  }

  EXPECT_EQ(cycle_enum::openmp::count_simple_cycles_johnson(view, 2),
            cycle_enum::sequential::count_simple_cycles_johnson(view));
}

TEST(OpenMPJohnsonTest, RejectsInvalidConfiguration) {
  const cycle_enum::GraphView view = representative_view();

  EXPECT_THROW((void)cycle_enum::openmp::count_simple_cycles_johnson(view, 0),
               std::invalid_argument);
  EXPECT_THROW((void)cycle_enum::openmp::count_simple_cycles_johnson(view, 1, 1),
               std::invalid_argument);
}

}  // namespace
