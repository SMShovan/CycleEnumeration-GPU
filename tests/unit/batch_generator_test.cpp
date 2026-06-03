#include "cycle_enum/core/graph.hpp"
#include "cycle_enum/core/graph_view.hpp"
#include "cycle_enum/dynamic/batch_generator.hpp"

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <unordered_set>
#include <vector>

#include <gtest/gtest.h>

namespace {

cycle_enum::GraphView ring_view(const std::size_t vertex_count) {
  // A directed ring plus a few chords, so there are both edges and non-edges.
  std::vector<cycle_enum::TemporalEdge> edges;
  for (std::size_t v = 0; v < vertex_count; ++v) {
    edges.push_back({static_cast<cycle_enum::VertexId>(v),
                     static_cast<cycle_enum::VertexId>((v + 1) % vertex_count),
                     {0}});
  }
  std::vector<cycle_enum::ExternalVertexId> external_ids;
  for (std::size_t v = 0; v < vertex_count; ++v) {
    external_ids.push_back(static_cast<cycle_enum::ExternalVertexId>(v));
  }
  return cycle_enum::build_graph_view(
      cycle_enum::TemporalGraph(std::move(external_ids), std::move(edges)));
}

std::unordered_set<std::uint64_t> edge_keys(const cycle_enum::GraphView& g) {
  std::unordered_set<std::uint64_t> keys;
  for (cycle_enum::VertexId u = 0; u < g.vertex_count(); ++u) {
    for (std::size_t off = g.outgoing_offsets()[u];
         off < g.outgoing_offsets()[u + 1]; ++off) {
      const cycle_enum::VertexId v = g.outgoing_edges()[off].vertex;
      keys.insert((static_cast<std::uint64_t>(u) << 32) | v);
    }
  }
  return keys;
}

TEST(BatchGeneratorTest, HonorsRequestedCounts) {
  const cycle_enum::GraphView view = ring_view(12);
  cycle_enum::dynamic::BatchParams params;
  params.num_deletions = 3;
  params.num_insertions = 4;
  params.seed = 42;

  const cycle_enum::dynamic::EdgeBatch batch =
      cycle_enum::dynamic::generate_batch(view, params);
  EXPECT_EQ(batch.deletions.size(), 3u);
  EXPECT_EQ(batch.insertions.size(), 4u);
}

TEST(BatchGeneratorTest, IsDeterministicForSeed) {
  const cycle_enum::GraphView view = ring_view(12);
  cycle_enum::dynamic::BatchParams params;
  params.num_deletions = 3;
  params.num_insertions = 3;
  params.seed = 7;

  const cycle_enum::dynamic::EdgeBatch a =
      cycle_enum::dynamic::generate_batch(view, params);
  const cycle_enum::dynamic::EdgeBatch b =
      cycle_enum::dynamic::generate_batch(view, params);
  EXPECT_EQ(a.deletions, b.deletions);
  EXPECT_EQ(a.insertions, b.insertions);
}

TEST(BatchGeneratorTest, DeletionsExistInsertionsDoNot) {
  const cycle_enum::GraphView view = ring_view(12);
  const std::unordered_set<std::uint64_t> keys = edge_keys(view);
  cycle_enum::dynamic::BatchParams params;
  params.num_deletions = 4;
  params.num_insertions = 5;
  params.seed = 99;

  const cycle_enum::dynamic::EdgeBatch batch =
      cycle_enum::dynamic::generate_batch(view, params);

  for (const auto& d : batch.deletions) {
    EXPECT_NE(keys.count((static_cast<std::uint64_t>(d.source) << 32) | d.target),
              0u);
  }
  for (const auto& i : batch.insertions) {
    EXPECT_NE(i.source, i.target);
    EXPECT_EQ(keys.count((static_cast<std::uint64_t>(i.source) << 32) | i.target),
              0u);
  }
}

TEST(BatchGeneratorTest, LocalityWindowConfinesEndpoints) {
  const cycle_enum::GraphView view = ring_view(40);
  cycle_enum::dynamic::BatchParams params;
  params.num_deletions = 2;
  params.num_insertions = 3;
  params.seed = 5;
  params.locality_window = 6;

  const cycle_enum::dynamic::EdgeBatch batch =
      cycle_enum::dynamic::generate_batch(view, params);

  cycle_enum::VertexId lo = 0xffffffffu;
  cycle_enum::VertexId hi = 0;
  const auto track = [&](cycle_enum::VertexId v) {
    lo = std::min(lo, v);
    hi = std::max(hi, v);
  };
  for (const auto& c : batch.deletions) {
    track(c.source);
    track(c.target);
  }
  for (const auto& c : batch.insertions) {
    track(c.source);
    track(c.target);
  }
  EXPECT_LT(hi - lo, 6u); // all endpoints lie within a window of size 6
}

TEST(BatchGeneratorTest, ThrowsWhenTooManyDeletions) {
  const cycle_enum::GraphView view = ring_view(8); // only 8 ring edges
  cycle_enum::dynamic::BatchParams params;
  params.num_deletions = 100;
  params.seed = 1;
  EXPECT_THROW((void)cycle_enum::dynamic::generate_batch(view, params),
               std::invalid_argument);
}

}  // namespace
