#include "cycle_enum/dynamic/edge_change.hpp"

#include <algorithm>

/**
 * @file edge_change.cpp
 * @brief Edge-change and batch normalization.
 */

namespace cycle_enum::dynamic {

void sort_and_dedup(std::vector<EdgeChange>& changes) {
  std::sort(changes.begin(), changes.end());
  changes.erase(std::unique(changes.begin(), changes.end()), changes.end());
}

void normalize(EdgeBatch& batch) {
  sort_and_dedup(batch.deletions);
  sort_and_dedup(batch.insertions);
}

}  // namespace cycle_enum::dynamic
