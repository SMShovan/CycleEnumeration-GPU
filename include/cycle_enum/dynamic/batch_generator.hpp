#pragma once

#include "cycle_enum/core/graph_view.hpp"
#include "cycle_enum/dynamic/edge_change.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>

/**
 * @file batch_generator.hpp
 * @brief Reproducible random edge-change batch generation.
 */

namespace cycle_enum::dynamic {

/**
 * @brief Parameters controlling random batch generation.
 */
struct BatchParams {
  std::size_t num_deletions = 0; ///< Number of existing edges to delete.
  std::size_t num_insertions = 0; ///< Number of new edges to insert.
  std::uint64_t seed = 0; ///< Seed for reproducible generation.

  /**
   * @brief Optional locality window.
   *
   * When set and smaller than the vertex count, both endpoints of every changed
   * edge are confined to a contiguous window of this many compact vertex ids,
   * chosen by the seed. This controls how local the changes are, which is the
   * main knob for sweeping the fraction of cycles a batch touches.
   */
  std::optional<std::size_t> locality_window = std::nullopt;
};

/**
 * @brief Generate a reproducible batch of edge changes for a graph.
 *
 * Deletions are sampled without replacement from existing directed edges;
 * insertions are sampled from directed pairs that are not edges and are not
 * self-loops. Because deletions come from the edge set and insertions from its
 * complement, the two are automatically disjoint. Generation is deterministic
 * for a given graph and `seed`.
 *
 * @param graph CSR/CSC graph view of the initial graph.
 * @param params Batch parameters.
 * @return A normalized batch (each list sorted, duplicate-free).
 *
 * @throws std::invalid_argument if the graph is too small, or there are not
 * enough existing edges or non-edges in the locality window to satisfy the
 * requested counts.
 */
[[nodiscard]] EdgeBatch generate_batch(const GraphView& graph,
                                       const BatchParams& params);

}  // namespace cycle_enum::dynamic
