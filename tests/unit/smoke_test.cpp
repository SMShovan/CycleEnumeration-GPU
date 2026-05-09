#include "cycle_enum/core/version.hpp"

#include <gtest/gtest.h>

namespace {

TEST(ProjectSmokeTest, ReportsConfiguredVersion) {
  const cycle_enum::Version current = cycle_enum::version();

  EXPECT_EQ(current.major, CYCLE_ENUM_VERSION_MAJOR);
  EXPECT_EQ(current.minor, CYCLE_ENUM_VERSION_MINOR);
  EXPECT_EQ(current.patch, CYCLE_ENUM_VERSION_PATCH);
  EXPECT_EQ(cycle_enum::version_string(), "0.1.0");
}

}  // namespace

