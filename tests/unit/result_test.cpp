#include "cycle_enum/core/result.hpp"

#include <gtest/gtest.h>

namespace {

TEST(ResultTest, StatusNamesAreStableForReportsAndCliOutput) {
  EXPECT_EQ(cycle_enum::to_string(cycle_enum::RunStatus::Success), "success");
  EXPECT_EQ(cycle_enum::to_string(cycle_enum::RunStatus::InvalidOptions),
            "invalid-options");
  EXPECT_EQ(cycle_enum::to_string(cycle_enum::RunStatus::Unsupported),
            "unsupported");
  EXPECT_EQ(cycle_enum::to_string(cycle_enum::RunStatus::RuntimeError),
            "runtime-error");
  EXPECT_EQ(cycle_enum::to_string(cycle_enum::RunStatus::Skipped), "skipped");
}

TEST(ResultTest, RuntimeStatisticsDefaultsToZero) {
  const cycle_enum::RuntimeStatistics stats;

  EXPECT_EQ(stats.graph_load_seconds, 0.0);
  EXPECT_EQ(stats.preprocessing_seconds, 0.0);
  EXPECT_EQ(stats.cpu_seconds, 0.0);
  EXPECT_EQ(stats.transfer_h2d_seconds, 0.0);
  EXPECT_EQ(stats.kernel_seconds, 0.0);
  EXPECT_EQ(stats.transfer_d2h_seconds, 0.0);
  EXPECT_EQ(stats.total_seconds, 0.0);
  EXPECT_EQ(stats.vertex_visits, 0U);
  EXPECT_EQ(stats.edge_visits, 0U);
  EXPECT_EQ(stats.start_work_items, 0U);
}

}  // namespace

