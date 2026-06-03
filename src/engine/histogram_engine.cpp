#include "cycle_enum/engine/histogram_engine.hpp"

#include "cycle_enum/dynamic/update_cuda.hpp"
#include "cycle_enum/dynamic/update_openmp.hpp"
#include "cycle_enum/dynamic/update_sequential.hpp"
#include "cycle_enum/sequential/johnson.hpp"

#include <stdexcept>

/**
 * @file histogram_engine.cpp
 * @brief Uniform entry points for full recomputation and incremental update.
 */

namespace cycle_enum::engine {

CycleHistogram count_histogram(const GraphView& graph,
                               const std::optional<std::size_t> max_cycle_length) {
  return sequential::count_simple_cycles_johnson(graph, max_cycle_length);
}

CycleHistogram update_histogram(const UpdateBackend backend,
                                const dynamic::DirectedGraph& initial_graph,
                                const CycleHistogram& prior,
                                const dynamic::EdgeBatch& batch,
                                const std::size_t max_cycle_length,
                                const int threads,
                                const int device_id) {
  switch (backend) {
    case UpdateBackend::Sequential:
      return dynamic::update_static_histogram(initial_graph, prior, batch,
                                              max_cycle_length);
    case UpdateBackend::OpenMP:
      return dynamic::update_static_histogram_openmp(
          initial_graph, prior, batch, max_cycle_length, threads);
    case UpdateBackend::Cuda:
      return dynamic::update_static_histogram_cuda(
          initial_graph, prior, batch, max_cycle_length, device_id);
  }
  throw std::logic_error("unsupported update backend");
}

}  // namespace cycle_enum::engine
