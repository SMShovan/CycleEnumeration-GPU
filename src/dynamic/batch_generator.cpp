#include "cycle_enum/dynamic/batch_generator.hpp"

#include <algorithm>
#include <random>
#include <stdexcept>
#include <unordered_set>
#include <vector>

/**
 * @file batch_generator.cpp
 * @brief Reproducible random edge-change batch generation.
 */

namespace cycle_enum::dynamic {

namespace {

std::uint64_t edge_key(const VertexId source, const VertexId target) noexcept {
  return (static_cast<std::uint64_t>(source) << 32) |
         static_cast<std::uint64_t>(target);
}

}  // namespace

EdgeBatch generate_batch(const GraphView& graph, const BatchParams& params) {
  EdgeBatch batch;
  if (params.num_deletions == 0 && params.num_insertions == 0) {
    return batch;
  }

  const std::size_t vertex_count = graph.vertex_count();
  if (vertex_count < 2) {
    throw std::invalid_argument(
        "batch generation requires at least two vertices");
  }

  std::unordered_set<std::uint64_t> existing;
  std::vector<EdgeChange> existing_edges;
  for (VertexId source = 0; source < vertex_count; ++source) {
    const std::size_t begin = graph.outgoing_offsets()[source];
    const std::size_t end = graph.outgoing_offsets()[source + 1];
    for (std::size_t offset = begin; offset < end; ++offset) {
      const VertexId target = graph.outgoing_edges()[offset].vertex;
      existing.insert(edge_key(source, target));
      existing_edges.push_back(EdgeChange{source, target});
    }
  }

  std::mt19937_64 rng(params.seed);

  VertexId window_lo = 0;
  VertexId window_hi = static_cast<VertexId>(vertex_count);
  if (params.locality_window.has_value() &&
      *params.locality_window < vertex_count) {
    const std::size_t window = std::max<std::size_t>(*params.locality_window, 2);
    if (window < vertex_count) {
      std::uniform_int_distribution<std::size_t> start_dist(
          0, vertex_count - window);
      const std::size_t start = start_dist(rng);
      window_lo = static_cast<VertexId>(start);
      window_hi = static_cast<VertexId>(start + window);
    }
  }
  const auto in_window = [&](const VertexId vertex) {
    return vertex >= window_lo && vertex < window_hi;
  };

  if (params.num_deletions > 0) {
    std::vector<EdgeChange> candidates;
    for (const EdgeChange& edge : existing_edges) {
      if (in_window(edge.source) && in_window(edge.target)) {
        candidates.push_back(edge);
      }
    }
    if (candidates.size() < params.num_deletions) {
      throw std::invalid_argument(
          "not enough existing edges in the locality window for the requested "
          "deletions");
    }
    std::shuffle(candidates.begin(), candidates.end(), rng);
    batch.deletions.assign(candidates.begin(),
                           candidates.begin() +
                               static_cast<std::ptrdiff_t>(params.num_deletions));
  }

  if (params.num_insertions > 0) {
    const std::size_t window = static_cast<std::size_t>(window_hi - window_lo);
    std::unordered_set<std::uint64_t> chosen;
    std::uniform_int_distribution<std::size_t> vertex_dist(
        window_lo, static_cast<std::size_t>(window_hi) - 1);
    const std::size_t attempt_cap =
        1000ull * (params.num_insertions + 1) + 100ull * window;
    std::size_t attempts = 0;
    while (batch.insertions.size() < params.num_insertions) {
      if (++attempts > attempt_cap) {
        throw std::invalid_argument(
            "could not sample enough non-edges in the locality window for the "
            "requested insertions");
      }
      const VertexId source = static_cast<VertexId>(vertex_dist(rng));
      const VertexId target = static_cast<VertexId>(vertex_dist(rng));
      if (source == target) {
        continue;
      }
      const std::uint64_t key = edge_key(source, target);
      if (existing.count(key) != 0 || chosen.count(key) != 0) {
        continue;
      }
      chosen.insert(key);
      batch.insertions.push_back(EdgeChange{source, target});
    }
  }

  normalize(batch);
  return batch;
}

}  // namespace cycle_enum::dynamic
