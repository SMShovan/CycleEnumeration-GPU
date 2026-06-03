#include "cycle_enum/core/graph.hpp"
#include "cycle_enum/core/graph_view.hpp"
#include "cycle_enum/dynamic/batch_generator.hpp"
#include "cycle_enum/dynamic/directed_graph.hpp"
#include "cycle_enum/dynamic/update_openmp.hpp"
#include "cycle_enum/dynamic/update_sequential.hpp"
#include "cycle_enum/openmp/openmp_config.hpp"
#include "cycle_enum/sequential/johnson.hpp"

#include <cstddef>
#include <random>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

namespace {

using cycle_enum::dynamic::DirectedGraph;

DirectedGraph random_graph(const std::size_t vertex_count,
                           const double edge_probability,
                           std::mt19937_64& rng) {
  std::vector<cycle_enum::TemporalEdge> edges;
  std::uniform_real_distribution<double> coin(0.0, 1.0);
  for (std::size_t u = 0; u < vertex_count; ++u) {
    for (std::size_t v = 0; v < vertex_count; ++v) {
      if (u != v && coin(rng) < edge_probability) {
        edges.push_back({static_cast<cycle_enum::VertexId>(u),
                         static_cast<cycle_enum::VertexId>(v), {0}});
      }
    }
  }
  std::vector<cycle_enum::ExternalVertexId> ids;
  for (std::size_t v = 0; v < vertex_count; ++v) {
    ids.push_back(static_cast<cycle_enum::ExternalVertexId>(v));
  }
  return cycle_enum::dynamic::build_directed_graph(cycle_enum::build_graph_view(
      cycle_enum::TemporalGraph(std::move(ids), std::move(edges))));
}

TEST(DynamicUpdateOpenMPTest, SingleThreadMatchesSequential) {
  std::mt19937_64 rng(321);
  constexpr std::size_t kMaxLength = 6;
  for (int trial = 0; trial < 30; ++trial) {
    const DirectedGraph g0 = random_graph(8, 0.35, rng);
    const cycle_enum::GraphView view0 = cycle_enum::dynamic::to_graph_view(g0);
    const cycle_enum::CycleHistogram h0 =
        cycle_enum::sequential::count_simple_cycles_johnson(view0, kMaxLength);

    cycle_enum::dynamic::BatchParams params;
    params.num_deletions = 2;
    params.num_insertions = 2;
    params.seed = rng();
    cycle_enum::dynamic::EdgeBatch batch;
    try {
      batch = cycle_enum::dynamic::generate_batch(view0, params);
    } catch (const std::invalid_argument&) {
      continue;
    }

    EXPECT_EQ(cycle_enum::dynamic::update_static_histogram_openmp(g0, h0, batch,
                                                                 kMaxLength, 1),
              cycle_enum::dynamic::update_static_histogram(g0, h0, batch,
                                                           kMaxLength));
  }
}

TEST(DynamicUpdateOpenMPTest, MultiThreadMatchesRecompute) {
  if (!cycle_enum::openmp::available()) {
    std::mt19937_64 rng(1);
    const DirectedGraph g0 = random_graph(6, 0.4, rng);
    const cycle_enum::CycleHistogram h0 =
        cycle_enum::sequential::count_simple_cycles_johnson(
            cycle_enum::dynamic::to_graph_view(g0), 6);
    EXPECT_THROW(
        (void)cycle_enum::dynamic::update_static_histogram_openmp(g0, h0, {}, 6,
                                                                  2),
        std::runtime_error);
    GTEST_SKIP() << "OpenMP runtime is not available in this build";
  }

  std::mt19937_64 rng(654);
  constexpr std::size_t kMaxLength = 6;
  for (int trial = 0; trial < 40; ++trial) {
    const DirectedGraph g0 = random_graph(8, 0.4, rng);
    const cycle_enum::GraphView view0 = cycle_enum::dynamic::to_graph_view(g0);
    const cycle_enum::CycleHistogram h0 =
        cycle_enum::sequential::count_simple_cycles_johnson(view0, kMaxLength);

    cycle_enum::dynamic::BatchParams params;
    params.num_deletions = 3;
    params.num_insertions = 3;
    params.seed = rng();
    cycle_enum::dynamic::EdgeBatch batch;
    try {
      batch = cycle_enum::dynamic::generate_batch(view0, params);
    } catch (const std::invalid_argument&) {
      continue;
    }

    const cycle_enum::CycleHistogram recomputed =
        cycle_enum::sequential::count_simple_cycles_johnson(
            cycle_enum::dynamic::to_graph_view(
                cycle_enum::dynamic::apply_batch(g0, batch)),
            kMaxLength);

    for (const int threads : {2, 4}) {
      EXPECT_EQ(cycle_enum::dynamic::update_static_histogram_openmp(
                    g0, h0, batch, kMaxLength, threads),
                recomputed)
          << "trial " << trial << " threads " << threads;
    }
  }
}

}  // namespace
