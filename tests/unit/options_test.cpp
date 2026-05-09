#include "cycle_enum/core/options.hpp"

#include <gtest/gtest.h>

namespace {

TEST(OptionsTest, DefaultsToSequentialJohnsonSimpleCycles) {
  const cycle_enum::CycleEnumerationOptions options =
      cycle_enum::default_options();

  EXPECT_EQ(options.algorithm, cycle_enum::AlgorithmFamily::Johnson);
  EXPECT_EQ(options.execution, cycle_enum::ExecutionPolicy::Sequential);
  EXPECT_EQ(options.mode, cycle_enum::CycleMode::Simple);
  EXPECT_FALSE(options.time_window.has_value());
  EXPECT_FALSE(options.max_cycle_length.has_value());
  EXPECT_EQ(options.openmp_threads, 1);
  EXPECT_EQ(options.cuda_device_id, 0);
  EXPECT_FALSE(options.use_cycle_union);
  EXPECT_FALSE(options.validate_results);
}

TEST(OptionsTest, NamesAreStableForReportsAndCliOutput) {
  EXPECT_EQ(cycle_enum::to_string(cycle_enum::CycleMode::Simple), "simple");
  EXPECT_EQ(cycle_enum::to_string(cycle_enum::CycleMode::SimpleTimeWindow),
            "simple-time-window");
  EXPECT_EQ(cycle_enum::to_string(cycle_enum::CycleMode::Temporal),
            "temporal");
  EXPECT_EQ(cycle_enum::to_string(cycle_enum::AlgorithmFamily::Johnson),
            "johnson");
  EXPECT_EQ(cycle_enum::to_string(cycle_enum::AlgorithmFamily::ReadTarjan),
            "read-tarjan");
  EXPECT_EQ(cycle_enum::to_string(cycle_enum::ExecutionPolicy::Cuda), "cuda");
}

TEST(OptionsTest, ValidatesTimeWindowModes) {
  cycle_enum::CycleEnumerationOptions options;
  options.mode = cycle_enum::CycleMode::Temporal;

  EXPECT_FALSE(cycle_enum::validate_options(options).empty());

  options.time_window = 3600;
  EXPECT_TRUE(cycle_enum::validate_options(options).empty());

  options.time_window = 0;
  EXPECT_FALSE(cycle_enum::validate_options(options).empty());
}

TEST(OptionsTest, ValidatesResourceControls) {
  cycle_enum::CycleEnumerationOptions options;
  options.openmp_threads = 0;
  options.cuda_device_id = -1;
  options.max_cycle_length = 1;

  const std::vector<std::string> errors =
      cycle_enum::validate_options(options);

  EXPECT_GE(errors.size(), 3U);
}

TEST(OptionsTest, RejectsCycleUnionOutsideTemporalMode) {
  cycle_enum::CycleEnumerationOptions options;
  options.use_cycle_union = true;

  EXPECT_FALSE(cycle_enum::validate_options(options).empty());

  options.mode = cycle_enum::CycleMode::Temporal;
  options.time_window = 42;
  EXPECT_TRUE(cycle_enum::validate_options(options).empty());
}

}  // namespace

