#include "cycle_enum/cuda/cuda_config.hpp"

#include <stdexcept>
#include <string>

#include <gtest/gtest.h>

TEST(CudaConfigTest, ReportsAvailabilityString) {
  EXPECT_FALSE(cycle_enum::cuda::availability_string().empty());
}

TEST(CudaConfigTest, HandlesDeviceQueriesByBuildAvailability) {
  if (!cycle_enum::cuda::compiled_with_cuda()) {
    EXPECT_EQ(cycle_enum::cuda::device_count(), 0);
    EXPECT_TRUE(cycle_enum::cuda::enumerate_devices().empty());
    EXPECT_THROW(cycle_enum::cuda::require_device(0), std::runtime_error);
    return;
  }

  const int count = cycle_enum::cuda::device_count();
  const std::vector<cycle_enum::cuda::CudaDeviceInfo> devices =
      cycle_enum::cuda::enumerate_devices();
  EXPECT_EQ(devices.size(), static_cast<std::size_t>(count));

  if (count > 0) {
    EXPECT_NO_THROW(cycle_enum::cuda::require_device(0));
  }
}

TEST(CudaConfigTest, RejectsNegativeDeviceIds) {
  EXPECT_THROW(cycle_enum::cuda::require_device(-1), std::invalid_argument);
}
