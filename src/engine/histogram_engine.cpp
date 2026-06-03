#include "cycle_enum/engine/histogram_engine.hpp"

#include "cycle_enum/sequential/johnson.hpp"

/**
 * @file histogram_engine.cpp
 * @brief Uniform entry points for full recomputation and incremental update.
 */

namespace cycle_enum::engine {

CycleHistogram count_histogram(const GraphView& graph,
                               const std::optional<std::size_t> max_cycle_length) {
  return sequential::count_simple_cycles_johnson(graph, max_cycle_length);
}

}  // namespace cycle_enum::engine
