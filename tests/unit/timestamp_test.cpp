#include "cycle_enum/core/timestamp.hpp"

#include <limits>
#include <stdexcept>

#include <gtest/gtest.h>

namespace {

TEST(TimestampTest, FindsInclusiveWindowRange) {
  const std::vector<cycle_enum::Timestamp> timestamps{1, 3, 3, 5, 8};

  const cycle_enum::TimestampRange range =
      cycle_enum::timestamps_in_window(timestamps, 3, 2);

  EXPECT_EQ(range.begin, 1U);
  EXPECT_EQ(range.end, 4U);
  EXPECT_EQ(range.size(), 3U);
  EXPECT_EQ(cycle_enum::first_timestamp(timestamps, range), 3);
}

TEST(TimestampTest, FindsStrictAfterStartWindowRange) {
  const std::vector<cycle_enum::Timestamp> timestamps{1, 3, 3, 5, 8};

  const cycle_enum::TimestampRange range = cycle_enum::timestamps_in_window(
      timestamps, 3, 2, cycle_enum::TimestampStartPolicy::AfterStart);

  EXPECT_EQ(range.begin, 3U);
  EXPECT_EQ(range.end, 4U);
  EXPECT_EQ(cycle_enum::first_timestamp(timestamps, range), 5);
}

TEST(TimestampTest, FindsWindowInsideSubrange) {
  const std::vector<cycle_enum::Timestamp> timestamps{1, 3, 5, 7, 9};

  const cycle_enum::TimestampRange range =
      cycle_enum::timestamps_in_window(timestamps, 1, 4, 4, 3);

  EXPECT_EQ(range.begin, 2U);
  EXPECT_EQ(range.end, 4U);
}

TEST(TimestampTest, ReportsEmptyWindow) {
  const std::vector<cycle_enum::Timestamp> timestamps{10, 20};

  EXPECT_FALSE(cycle_enum::has_timestamp_in_window(timestamps, 11, 8));
  EXPECT_TRUE(cycle_enum::timestamps_in_window(timestamps, 11, 8).empty());
}

TEST(TimestampTest, FindsTemporalStrictlyIncreasingRange) {
  const std::vector<cycle_enum::Timestamp> timestamps{1, 2, 2, 4, 7};

  const cycle_enum::TimestampRange range =
      cycle_enum::timestamps_after(timestamps, 2, 6);

  EXPECT_EQ(range.begin, 3U);
  EXPECT_EQ(range.end, 4U);
  EXPECT_EQ(cycle_enum::first_timestamp(timestamps, range), 4);
}

TEST(TimestampTest, FindsTemporalRangeInsideSubrange) {
  const std::vector<cycle_enum::Timestamp> timestamps{1, 2, 4, 6, 8};

  const cycle_enum::TimestampRange range =
      cycle_enum::timestamps_after(timestamps, 1, 4, 2, 6);

  EXPECT_EQ(range.begin, 2U);
  EXPECT_EQ(range.end, 4U);
}

TEST(TimestampTest, RejectsInvalidWindowWidth) {
  const std::vector<cycle_enum::Timestamp> timestamps{1, 2};

  EXPECT_THROW((void)cycle_enum::timestamps_in_window(timestamps, 1, -1),
               std::invalid_argument);
}

TEST(TimestampTest, DetectsWindowEndOverflow) {
  const std::vector<cycle_enum::Timestamp> timestamps{1, 2};

  EXPECT_THROW(
      (void)cycle_enum::timestamps_in_window(
          timestamps, std::numeric_limits<cycle_enum::Timestamp>::max(), 1),
      std::overflow_error);
}

TEST(TimestampTest, FirstTimestampReturnsEmptyForInvalidRange) {
  const std::vector<cycle_enum::Timestamp> timestamps{1};

  EXPECT_FALSE(
      cycle_enum::first_timestamp(timestamps, cycle_enum::TimestampRange{4, 9})
          .has_value());
}

}  // namespace
