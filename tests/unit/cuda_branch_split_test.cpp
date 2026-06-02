#include "cycle_enum/core/graph.hpp"
#include "cycle_enum/core/graph_view.hpp"
#include "cycle_enum/core/histogram.hpp"
#include "cycle_enum/cuda/cuda_branch_split.hpp"
#include "cycle_enum/sequential/johnson.hpp"

#include <cstddef>
#include <stdexcept>
#include <functional>
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

// Independent host model of the device continuation: from a prefix, depth-first
// search deeper and count every closure back to the root.
void continue_from_prefix(const cycle_enum::GraphView& graph,
                          const cycle_enum::VertexId root,
                          std::vector<cycle_enum::VertexId>& path,
                          std::vector<unsigned char>& visited,
                          cycle_enum::CycleHistogram& histogram) {
  const cycle_enum::VertexId current = path.back();
  const std::size_t begin = graph.outgoing_offsets()[current];
  const std::size_t end = graph.outgoing_offsets()[current + 1];
  for (std::size_t offset = begin; offset < end; ++offset) {
    const cycle_enum::VertexId next = graph.outgoing_edges()[offset].vertex;
    if (next == root && path.size() >= 2) {
      histogram.increment(
          static_cast<cycle_enum::CycleHistogram::Length>(path.size()));
      continue;
    }
    if (next <= root || visited[next] != 0) {
      continue;
    }
    visited[next] = 1;
    path.push_back(next);
    continue_from_prefix(graph, root, path, visited, histogram);
    path.pop_back();
    visited[next] = 0;
  }
}

cycle_enum::CycleHistogram recombine(const cycle_enum::GraphView& graph,
                                     const cycle_enum::cuda::BranchSplit& split) {
  cycle_enum::CycleHistogram total = split.closed;
  for (const cycle_enum::cuda::SplitWorkItem& item : split.items) {
    std::vector<cycle_enum::VertexId> path = item.prefix;
    std::vector<unsigned char> visited(graph.vertex_count(), 0);
    for (const cycle_enum::VertexId vertex : item.prefix) {
      visited[vertex] = 1;
    }
    cycle_enum::CycleHistogram continued;
    continue_from_prefix(graph, item.prefix.front(), path, visited, continued);
    total.merge(continued);
  }
  return total;
}

cycle_enum::GraphView dense_fixture() {
  return view_from_edges(
      {
          {0, 1, {0}}, {1, 0, {0}}, {1, 2, {0}}, {2, 0, {0}},
          {2, 3, {0}}, {3, 1, {0}}, {0, 3, {0}}, {3, 0, {0}},
          {2, 1, {0}}, {0, 2, {0}}, {3, 2, {0}},
      },
      4);
}

TEST(CudaBranchSplitTest, RecombinesToUndecomposedCount) {
  const cycle_enum::GraphView view = dense_fixture();
  const cycle_enum::CycleHistogram expected =
      cycle_enum::sequential::count_simple_cycles_johnson(view);

  for (const std::size_t cutoff : {std::size_t{2}, std::size_t{3},
                                   std::size_t{4}, std::size_t{5}}) {
    const cycle_enum::cuda::BranchSplit split =
        cycle_enum::cuda::split_static_search(view, cutoff);
    EXPECT_EQ(recombine(view, split), expected) << "cutoff=" << cutoff;
  }
}

TEST(CudaBranchSplitTest, RespectsMaxCycleLength) {
  const cycle_enum::GraphView view = dense_fixture();
  const cycle_enum::cuda::BranchSplit split =
      cycle_enum::cuda::split_static_search(view, 2, std::size_t{3});
  // With a max length of 3, recombination must still match a bounded reference.
  cycle_enum::CycleHistogram total = split.closed;
  for (const cycle_enum::cuda::SplitWorkItem& item : split.items) {
    std::vector<cycle_enum::VertexId> path = item.prefix;
    std::vector<unsigned char> visited(view.vertex_count(), 0);
    for (const cycle_enum::VertexId vertex : item.prefix) {
      visited[vertex] = 1;
    }
    // Reference continuation bounded to length 3.
    cycle_enum::CycleHistogram continued;
    std::function<void(cycle_enum::VertexId)> dfs =
        [&](const cycle_enum::VertexId current) {
          const std::size_t begin = view.outgoing_offsets()[current];
          const std::size_t end = view.outgoing_offsets()[current + 1];
          for (std::size_t offset = begin; offset < end; ++offset) {
            const cycle_enum::VertexId next =
                view.outgoing_edges()[offset].vertex;
            if (next == item.prefix.front() && path.size() >= 2) {
              continued.increment(
                  static_cast<cycle_enum::CycleHistogram::Length>(path.size()));
              continue;
            }
            if (next <= item.prefix.front() || visited[next] != 0) {
              continue;
            }
            if (path.size() >= 3) {
              continue;
            }
            visited[next] = 1;
            path.push_back(next);
            dfs(next);
            path.pop_back();
            visited[next] = 0;
          }
        };
    dfs(path.back());
    total.merge(continued);
  }
  EXPECT_EQ(total,
            cycle_enum::sequential::count_simple_cycles_johnson(view, std::size_t{3}));
}

TEST(CudaBranchSplitTest, RejectsInvalidConfiguration) {
  const cycle_enum::GraphView view = dense_fixture();
  EXPECT_THROW((void)cycle_enum::cuda::split_static_search(view, 1),
               std::invalid_argument);
  EXPECT_THROW(
      (void)cycle_enum::cuda::split_static_search(view, 2, std::size_t{1}),
      std::invalid_argument);
}

}  // namespace
