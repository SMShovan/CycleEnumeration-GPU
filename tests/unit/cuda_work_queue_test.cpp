#include "cycle_enum/core/graph.hpp"
#include "cycle_enum/core/graph_view.hpp"
#include "cycle_enum/cuda/cuda_config.hpp"
#include "cycle_enum/cuda/cuda_work_queue.hpp"

#include <cstdlib>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

namespace {

void clear_tuning_env() {
  unsetenv("CYCLE_ENUM_CUDA_BLOCK_SIZE");
  unsetenv("CYCLE_ENUM_CUDA_BLOCKS_PER_SM");
}

TEST(CudaWorkQueueTest, TuningDefaultsWithoutEnvironment) {
  clear_tuning_env();
  const cycle_enum::cuda::WorkQueueTuning tuning =
      cycle_enum::cuda::work_queue_tuning_from_env();
  EXPECT_EQ(tuning.block_size, 128u);
  EXPECT_EQ(tuning.blocks_per_sm, 16u);
}

TEST(CudaWorkQueueTest, TuningHonorsEnvironmentOverrides) {
  setenv("CYCLE_ENUM_CUDA_BLOCK_SIZE", "256", 1);
  setenv("CYCLE_ENUM_CUDA_BLOCKS_PER_SM", "8", 1);
  const cycle_enum::cuda::WorkQueueTuning tuning =
      cycle_enum::cuda::work_queue_tuning_from_env();
  EXPECT_EQ(tuning.block_size, 256u);
  EXPECT_EQ(tuning.blocks_per_sm, 8u);
  clear_tuning_env();
}

TEST(CudaWorkQueueTest, TuningRejectsInvalidEnvironment) {
  setenv("CYCLE_ENUM_CUDA_BLOCK_SIZE", "100", 1); // not a multiple of 32
  EXPECT_THROW((void)cycle_enum::cuda::work_queue_tuning_from_env(),
               std::invalid_argument);

  setenv("CYCLE_ENUM_CUDA_BLOCK_SIZE", "abc", 1);
  EXPECT_THROW((void)cycle_enum::cuda::work_queue_tuning_from_env(),
               std::invalid_argument);

  setenv("CYCLE_ENUM_CUDA_BLOCK_SIZE", "0", 1);
  EXPECT_THROW((void)cycle_enum::cuda::work_queue_tuning_from_env(),
               std::invalid_argument);
  clear_tuning_env();
}

TEST(CudaWorkQueueTest, PlanFillsDeviceButNotBeyondWork) {
  // Resident capacity is blocks_per_sm * sm_count = 16 * 10 = 160.
  const cycle_enum::cuda::WorkQueueLaunch many =
      cycle_enum::cuda::plan_work_queue_launch(1000, 128, 16, 10);
  EXPECT_EQ(many.block_size, 128u);
  EXPECT_EQ(many.grid_blocks, 160u);

  // Fewer work items than resident capacity caps the grid at the work count.
  const cycle_enum::cuda::WorkQueueLaunch few =
      cycle_enum::cuda::plan_work_queue_launch(5, 128, 16, 10);
  EXPECT_EQ(few.grid_blocks, 5u);
}

TEST(CudaWorkQueueTest, PlanIsEmptyForNoWork) {
  const cycle_enum::cuda::WorkQueueLaunch launch =
      cycle_enum::cuda::plan_work_queue_launch(0, 128, 16, 10);
  EXPECT_EQ(launch.grid_blocks, 0u);
  EXPECT_EQ(launch.block_size, 128u);
}

TEST(CudaWorkQueueTest, PlanRejectsZeroConfiguration) {
  EXPECT_THROW((void)cycle_enum::cuda::plan_work_queue_launch(10, 0, 16, 10),
               std::invalid_argument);
  EXPECT_THROW((void)cycle_enum::cuda::plan_work_queue_launch(10, 128, 0, 10),
               std::invalid_argument);
  EXPECT_THROW((void)cycle_enum::cuda::plan_work_queue_launch(10, 128, 16, 0),
               std::invalid_argument);
}

TEST(CudaWorkQueueTest, CounterRejectsTinyMaxCycleLength) {
  const cycle_enum::GraphView view = cycle_enum::build_graph_view(
      cycle_enum::TemporalGraph({0, 1}, {{0, 1, {0}}, {1, 0, {0}}}));
  EXPECT_THROW(
      (void)cycle_enum::cuda::count_simple_cycles_johnson_work_queue(view, 0, 1),
      std::invalid_argument);
}

TEST(CudaWorkQueueTest, CounterRequiresCompiledCudaBackend) {
  if (cycle_enum::cuda::compiled_with_cuda()) {
    GTEST_SKIP() << "CUDA backend is compiled; device parity runs on the GPU";
  }
  const cycle_enum::GraphView view = cycle_enum::build_graph_view(
      cycle_enum::TemporalGraph({0, 1}, {{0, 1, {0}}, {1, 0, {0}}}));
  EXPECT_THROW(
      (void)cycle_enum::cuda::count_simple_cycles_johnson_work_queue(view, 0, 4),
      std::runtime_error);
}

}  // namespace
