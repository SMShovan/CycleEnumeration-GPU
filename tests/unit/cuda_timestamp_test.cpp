#include "cycle_enum/cuda/cuda_timestamp.hpp"

#include <array>
#include <limits>

#include <gtest/gtest.h>

namespace {

TEST(CudaTimestampTest, BinarySearchMatchesDuplicateBoundaries) {
  constexpr std::array<cycle_enum::Timestamp, 6> timestamps{1, 3, 3, 3, 7, 9};

  EXPECT_EQ(cycle_enum::cuda::timestamp_lower_bound(
                timestamps.data(), 0, timestamps.size(),
                cycle_enum::Timestamp{3}),
            1U);
  EXPECT_EQ(cycle_enum::cuda::timestamp_upper_bound(
                timestamps.data(), 0, timestamps.size(),
                cycle_enum::Timestamp{3}),
            4U);
}

TEST(CudaTimestampTest, FindsInclusiveWindowInsideSubrange) {
  constexpr std::array<cycle_enum::Timestamp, 6> timestamps{1, 3, 3, 5, 8, 13};

  const cycle_enum::cuda::CudaTimestampRange range =
      cycle_enum::cuda::timestamps_in_window(timestamps.data(), 1, 5, 3, 2);

  EXPECT_EQ(range.begin, 1U);
  EXPECT_EQ(range.end, 4U);
  EXPECT_EQ(range.size(), 3U);
}

TEST(CudaTimestampTest, SupportsStrictAfterStartWindowPolicy) {
  constexpr std::array<cycle_enum::Timestamp, 5> timestamps{1, 3, 3, 5, 8};

  const cycle_enum::cuda::CudaTimestampRange range =
      cycle_enum::cuda::timestamps_in_window(
          timestamps.data(), 0, timestamps.size(), 3, 2,
          cycle_enum::cuda::CudaTimestampStartPolicy::AfterStart);

  EXPECT_EQ(range.begin, 3U);
  EXPECT_EQ(range.end, 4U);
  EXPECT_TRUE(cycle_enum::cuda::has_timestamp_in_window(
      timestamps.data(), 0, timestamps.size(), 3, 2,
      cycle_enum::cuda::CudaTimestampStartPolicy::AfterStart));
}

TEST(CudaTimestampTest, FindsTemporalStrictlyIncreasingRange) {
  constexpr std::array<cycle_enum::Timestamp, 5> timestamps{1, 2, 2, 4, 7};

  const cycle_enum::cuda::CudaTimestampRange range =
      cycle_enum::cuda::timestamps_after(timestamps.data(), 0,
                                         timestamps.size(), 2, 6);

  EXPECT_EQ(range.begin, 3U);
  EXPECT_EQ(range.end, 4U);
}

TEST(CudaTimestampTest, InvalidWindowReturnsEmptyRange) {
  constexpr std::array<cycle_enum::Timestamp, 2> timestamps{1, 2};

  const cycle_enum::cuda::CudaTimestampRange negative_width =
      cycle_enum::cuda::timestamps_in_window(timestamps.data(), 0,
                                             timestamps.size(), 1, -1);
  EXPECT_TRUE(negative_width.empty());

  const cycle_enum::cuda::CudaTimestampRange overflow =
      cycle_enum::cuda::timestamps_in_window(
          timestamps.data(), 0, timestamps.size(),
          std::numeric_limits<cycle_enum::Timestamp>::max(), 1);
  EXPECT_TRUE(overflow.empty());
}

}  // namespace
