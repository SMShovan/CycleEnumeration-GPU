#include "cycle_enum/core/graph.hpp"
#include "cycle_enum/core/graph_view.hpp"
#include "cycle_enum/cuda/cuda_work_item.hpp"

#include <algorithm>
#include <cstddef>
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

// Independent host reference: does a temporal cycle exist from this start event
// using strictly increasing timestamps inside the start window? Used to confirm
// the prefilter never drops a start event that can produce a cycle.
bool extends_to_cycle(const cycle_enum::GraphView& graph,
                      const cycle_enum::VertexId root,
                      const cycle_enum::VertexId current,
                      const cycle_enum::Timestamp last_timestamp,
                      const cycle_enum::Timestamp window_end,
                      std::vector<unsigned char>& visited) {
  const std::size_t begin = graph.outgoing_offsets()[current];
  const std::size_t end = graph.outgoing_offsets()[current + 1];
  for (std::size_t offset = begin; offset < end; ++offset) {
    const cycle_enum::AdjacencyEntry& edge = graph.outgoing_edges()[offset];
    for (std::size_t t = edge.timestamp_begin; t < edge.timestamp_end; ++t) {
      const cycle_enum::Timestamp timestamp = graph.timestamps()[t];
      if (timestamp <= last_timestamp || timestamp > window_end) {
        continue;
      }
      if (edge.vertex == root) {
        return true;
      }
      if (visited[edge.vertex] != 0) {
        continue;
      }
      visited[edge.vertex] = 1;
      if (extends_to_cycle(graph, root, edge.vertex, timestamp, window_end,
                           visited)) {
        return true;
      }
      visited[edge.vertex] = 0;
    }
  }
  return false;
}

bool has_temporal_cycle(const cycle_enum::GraphView& graph,
                        const cycle_enum::cuda::CudaStartEvent& event,
                        const cycle_enum::Timestamp window_width) {
  std::vector<unsigned char> visited(graph.vertex_count(), 0);
  visited[event.root] = 1;
  visited[event.first_vertex] = 1;
  return extends_to_cycle(graph, event.root, event.first_vertex,
                          event.start_timestamp,
                          event.start_timestamp + window_width, visited);
}

// Two live start edges (0->1->0 and 3->2->3) and two dead ones whose only
// return edge is before the start timestamp (1->0->1 and 2->3->2).
cycle_enum::GraphView mixed_fixture() {
  return view_from_edges(
      {
          {0, 1, {10}},
          {1, 0, {20}},
          {2, 3, {5}},
          {3, 2, {1}},
      },
      4);
}

TEST(CudaWorkItemTest, BuildsOneEventPerTimestamp) {
  const cycle_enum::GraphView view = mixed_fixture();
  const std::vector<cycle_enum::cuda::CudaStartEvent> events =
      cycle_enum::cuda::build_start_events(view);
  EXPECT_EQ(events.size(), view.timestamp_count());
}

TEST(CudaWorkItemTest, PrefilterKeepsAllWithoutCycleUnion) {
  const cycle_enum::GraphView view = mixed_fixture();
  const cycle_enum::cuda::StartEventSet set =
      cycle_enum::cuda::build_temporal_start_events(view, 100, false);
  EXPECT_EQ(set.generated, view.timestamp_count());
  EXPECT_EQ(set.events.size(), view.timestamp_count());
  EXPECT_EQ(set.dropped_by_cycle_union, 0u);
}

TEST(CudaWorkItemTest, PrefilterDropsDeadStartEdges) {
  const cycle_enum::GraphView view = mixed_fixture();
  const cycle_enum::cuda::StartEventSet set =
      cycle_enum::cuda::build_temporal_start_events(view, 100, true);

  EXPECT_EQ(set.generated, 4u);
  EXPECT_GT(set.dropped_by_cycle_union, 0u);
  EXPECT_EQ(set.events.size() + set.dropped_by_cycle_union, set.generated);
}

TEST(CudaWorkItemTest, PrefilterNeverDropsCyclicStartEvents) {
  const cycle_enum::GraphView view = mixed_fixture();
  const cycle_enum::Timestamp window = 100;

  const std::vector<cycle_enum::cuda::CudaStartEvent> all =
      cycle_enum::cuda::build_start_events(view);
  const cycle_enum::cuda::StartEventSet kept =
      cycle_enum::cuda::build_temporal_start_events(view, window, true);

  // Every start event that can actually close a cycle must survive the filter.
  for (const cycle_enum::cuda::CudaStartEvent& event : all) {
    if (!has_temporal_cycle(view, event, window)) {
      continue;
    }
    const bool present =
        std::any_of(kept.events.begin(), kept.events.end(),
                    [&](const cycle_enum::cuda::CudaStartEvent& candidate) {
                      return candidate.root == event.root &&
                             candidate.first_vertex == event.first_vertex &&
                             candidate.start_timestamp == event.start_timestamp;
                    });
    EXPECT_TRUE(present)
        << "prefilter dropped a cyclic start event rooted at " << event.root;
  }
}

TEST(CudaWorkItemTest, RejectsNegativeWindow) {
  const cycle_enum::GraphView view = mixed_fixture();
  EXPECT_THROW(
      (void)cycle_enum::cuda::build_temporal_start_events(view, -1, true),
      std::invalid_argument);
}

}  // namespace
