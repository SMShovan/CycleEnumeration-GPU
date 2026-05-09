#include "cycle_enum/openmp/openmp_config.hpp"

#include <stdexcept>

#include <gtest/gtest.h>

namespace {

TEST(OpenMPConfigTest, ReportsConsistentAvailability) {
  const cycle_enum::openmp::OpenMPConfig config =
      cycle_enum::openmp::config();

  EXPECT_EQ(cycle_enum::openmp::available(), config.available);
  EXPECT_EQ(cycle_enum::openmp::max_threads(), config.max_threads);
  EXPECT_GE(config.max_threads, 1);
  EXPECT_FALSE(cycle_enum::openmp::availability_string().empty());
}

TEST(OpenMPConfigTest, ResolvesSingleThreadWithoutRuntime) {
  EXPECT_EQ(cycle_enum::openmp::resolve_thread_count(1), 1);
  EXPECT_THROW((void)cycle_enum::openmp::resolve_thread_count(0),
               std::invalid_argument);
}

TEST(OpenMPConfigTest, HandlesMultiThreadRequestByAvailability) {
  if (!cycle_enum::openmp::available()) {
    EXPECT_THROW((void)cycle_enum::openmp::resolve_thread_count(2),
                 std::runtime_error);
    GTEST_SKIP() << "OpenMP runtime is not available in this build";
  }

  EXPECT_EQ(cycle_enum::openmp::resolve_thread_count(2), 2);
}

}  // namespace
