#include "cycle_enum/core/graph.hpp"
#include "cycle_enum/core/graph_view.hpp"
#include "cycle_enum/dynamic/batch_generator.hpp"
#include "cycle_enum/dynamic/directed_graph.hpp"
#include "cycle_enum/dynamic/update_sequential.hpp"
#include "cycle_enum/sequential/johnson.hpp"

#include <cstddef>
#include <random>
#include <set>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

namespace {

using cycle_enum::dynamic::DirectedGraph;

// Build a random directed graph (no self-loops) as a DirectedGraph.
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
  std::vector<cycle_enum::ExternalVertexId> external_ids;
  for (std::size_t v = 0; v < vertex_count; ++v) {
    external_ids.push_back(static_cast<cycle_enum::ExternalVertexId>(v));
  }
  return cycle_enum::dynamic::build_directed_graph(cycle_enum::build_graph_view(
      cycle_enum::TemporalGraph(std::move(external_ids), std::move(edges))));
}

std::size_t edge_count(const DirectedGraph& g) { return g.neighbors.size(); }

// The golden gate: for many random graphs and batches, the incremental update
// must equal a full recomputation of the post-batch graph.
TEST(DynamicUpdateParityTest, UpdateMatchesRecompute) {
  constexpr std::size_t kVertexCount = 8;
  constexpr std::size_t kMaxLength = 6;
  std::mt19937_64 rng(12345);

  int checked = 0;
  for (int trial = 0; trial < 200; ++trial) {
    const DirectedGraph g0 = random_graph(kVertexCount, 0.35, rng);
    const cycle_enum::GraphView view0 = cycle_enum::dynamic::to_graph_view(g0);
    const cycle_enum::CycleHistogram h0 =
        cycle_enum::sequential::count_simple_cycles_johnson(view0, kMaxLength);

    std::uniform_int_distribution<std::size_t> small(0, 3);
    cycle_enum::dynamic::BatchParams params;
    params.num_deletions = std::min(small(rng), edge_count(g0));
    params.num_insertions = small(rng);
    params.seed = rng();

    cycle_enum::dynamic::EdgeBatch batch;
    try {
      batch = cycle_enum::dynamic::generate_batch(view0, params);
    } catch (const std::invalid_argument&) {
      continue; // not enough candidates for this random graph; skip
    }

    const cycle_enum::CycleHistogram updated =
        cycle_enum::dynamic::update_static_histogram(g0, h0, batch, kMaxLength);

    const DirectedGraph g_final =
        cycle_enum::dynamic::apply_batch(g0, batch);
    const cycle_enum::CycleHistogram recomputed =
        cycle_enum::sequential::count_simple_cycles_johnson(
            cycle_enum::dynamic::to_graph_view(g_final), kMaxLength);

    ASSERT_EQ(updated, recomputed)
        << "trial " << trial << " deletions=" << batch.deletions.size()
        << " insertions=" << batch.insertions.size();
    ++checked;
  }
  EXPECT_GT(checked, 100); // most trials should produce a usable batch
}

TEST(DynamicUpdateParityTest, EmptyBatchLeavesHistogramUnchanged) {
  std::mt19937_64 rng(7);
  const DirectedGraph g0 = random_graph(7, 0.4, rng);
  const cycle_enum::CycleHistogram h0 =
      cycle_enum::sequential::count_simple_cycles_johnson(
          cycle_enum::dynamic::to_graph_view(g0), 6);

  const cycle_enum::CycleHistogram updated =
      cycle_enum::dynamic::update_static_histogram(g0, h0, {}, 6);
  EXPECT_EQ(updated, h0);
}

TEST(DynamicUpdateParityTest, AllDeleteAndAllInsertMatchRecompute) {
  std::mt19937_64 rng(99);
  const DirectedGraph g0 = random_graph(8, 0.4, rng);
  const cycle_enum::GraphView view0 = cycle_enum::dynamic::to_graph_view(g0);
  const cycle_enum::CycleHistogram h0 =
      cycle_enum::sequential::count_simple_cycles_johnson(view0, 6);

  cycle_enum::dynamic::BatchParams del_params;
  del_params.num_deletions = 3;
  del_params.seed = 1;
  const cycle_enum::dynamic::EdgeBatch del_batch =
      cycle_enum::dynamic::generate_batch(view0, del_params);
  EXPECT_EQ(cycle_enum::dynamic::update_static_histogram(g0, h0, del_batch, 6),
            cycle_enum::sequential::count_simple_cycles_johnson(
                cycle_enum::dynamic::to_graph_view(
                    cycle_enum::dynamic::apply_batch(g0, del_batch)),
                6));

  cycle_enum::dynamic::BatchParams ins_params;
  ins_params.num_insertions = 4;
  ins_params.seed = 2;
  const cycle_enum::dynamic::EdgeBatch ins_batch =
      cycle_enum::dynamic::generate_batch(view0, ins_params);
  EXPECT_EQ(cycle_enum::dynamic::update_static_histogram(g0, h0, ins_batch, 6),
            cycle_enum::sequential::count_simple_cycles_johnson(
                cycle_enum::dynamic::to_graph_view(
                    cycle_enum::dynamic::apply_batch(g0, ins_batch)),
                6));
}

}  // namespace
