#include "cycle_enum/core/graph.hpp"
#include "cycle_enum/core/graph_view.hpp"
#include "cycle_enum/cuda/cuda_config.hpp"
#include "cycle_enum/cuda/cuda_johnson.hpp"
#include "cycle_enum/sequential/johnson.hpp"

#include <stdexcept>

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

cycle_enum::GraphView representative_view() {
  return view_from_edges(
      {
          {0, 1, {0}},
          {1, 2, {0}},
          {2, 0, {0}},
          {1, 3, {0}},
          {3, 0, {0}},
      },
      4);
}

}  // namespace

TEST(CudaJohnsonTest, RejectsInvalidMaximumLength) {
  const cycle_enum::GraphView view = representative_view();

  EXPECT_THROW(
      (void)cycle_enum::cuda::count_simple_cycles_johnson(view, 0, 1),
      std::invalid_argument);
}

TEST(CudaJohnsonTest, RequiresCompiledCudaBackend) {
  const cycle_enum::GraphView view = representative_view();

  if (!cycle_enum::cuda::compiled_with_cuda()) {
    EXPECT_THROW(
        (void)cycle_enum::cuda::count_simple_cycles_johnson(view, 0, 4),
        std::runtime_error);
    GTEST_SKIP() << "CUDA support is not compiled in this build";
  }

  if (cycle_enum::cuda::device_count() == 0) {
    EXPECT_THROW(
        (void)cycle_enum::cuda::count_simple_cycles_johnson(view, 0, 4),
        std::out_of_range);
    GTEST_SKIP() << "No CUDA device is visible in this build";
  }

  EXPECT_EQ(cycle_enum::cuda::count_simple_cycles_johnson(view, 0, 4),
            cycle_enum::sequential::count_simple_cycles_johnson(view, 4));
}
