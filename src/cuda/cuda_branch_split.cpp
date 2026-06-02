#include "cycle_enum/cuda/cuda_branch_split.hpp"

#include <stdexcept>

/**
 * @file cuda_branch_split.cpp
 * @brief Host-side branch splitting for balanced CUDA cycle work items.
 *
 * The decomposition is host-only so it can be unit tested without a CUDA
 * device. The boundary rule is deliberately simple: the host counts every cycle
 * shorter than the cutoff, and every path prefix of exactly the cutoff length
 * becomes a continuation work item whose deeper cycles (length >= cutoff) are
 * found by the device. There is no asymmetric closure handling at the boundary,
 * so the device contract is just "depth-first search from the prefix, counting
 * each closure".
 */

namespace cycle_enum::cuda {

namespace {

bool prefix_has_work(const GraphView& graph,
                     const VertexId root,
                     const std::vector<VertexId>& path,
                     const std::vector<unsigned char>& visited,
                     const std::optional<std::size_t> max_cycle_length) {
  const VertexId current = path.back();
  const std::size_t begin = graph.outgoing_offsets()[current];
  const std::size_t end = graph.outgoing_offsets()[current + 1];
  for (std::size_t offset = begin; offset < end; ++offset) {
    const VertexId next = graph.outgoing_edges()[offset].vertex;
    if (next == root && path.size() >= 2) {
      return true; // device will count this closure at the cutoff length
    }
    if (next <= root || visited[next] != 0) {
      continue;
    }
    if (max_cycle_length.has_value() && path.size() >= *max_cycle_length) {
      continue;
    }
    return true; // a deeper continuation exists
  }
  return false;
}

void expand(const GraphView& graph,
            const VertexId root,
            const std::size_t cutoff_depth,
            const std::optional<std::size_t> max_cycle_length,
            std::vector<VertexId>& path,
            std::vector<unsigned char>& visited,
            BranchSplit& out) {
  if (path.size() == cutoff_depth) {
    if (prefix_has_work(graph, root, path, visited, max_cycle_length)) {
      out.items.push_back(SplitWorkItem{path});
    }
    return;
  }

  const VertexId current = path.back();
  const std::size_t begin = graph.outgoing_offsets()[current];
  const std::size_t end = graph.outgoing_offsets()[current + 1];
  for (std::size_t offset = begin; offset < end; ++offset) {
    const VertexId next = graph.outgoing_edges()[offset].vertex;

    if (next == root && path.size() >= 2) {
      out.closed.increment(static_cast<CycleHistogram::Length>(path.size()));
      continue;
    }

    if (next <= root || visited[next] != 0) {
      continue;
    }

    if (max_cycle_length.has_value() && path.size() >= *max_cycle_length) {
      continue;
    }

    visited[next] = 1;
    path.push_back(next);
    expand(graph, root, cutoff_depth, max_cycle_length, path, visited, out);
    path.pop_back();
    visited[next] = 0;
  }
}

}  // namespace

BranchSplit split_static_search(
    const GraphView& graph,
    const std::size_t cutoff_depth,
    const std::optional<std::size_t> max_cycle_length) {
  if (cutoff_depth < 2) {
    throw std::invalid_argument("cutoff_depth must be at least 2");
  }
  if (max_cycle_length.has_value() && *max_cycle_length < 2) {
    throw std::invalid_argument("max_cycle_length must be at least 2");
  }

  BranchSplit out;
  std::vector<VertexId> path;
  std::vector<unsigned char> visited;
  path.reserve(cutoff_depth);

  for (VertexId root = 0; root < graph.vertex_count(); ++root) {
    visited.assign(graph.vertex_count(), 0);
    path.clear();
    path.push_back(root);
    visited[root] = 1;
    expand(graph, root, cutoff_depth, max_cycle_length, path, visited, out);
  }

  return out;
}

}  // namespace cycle_enum::cuda
