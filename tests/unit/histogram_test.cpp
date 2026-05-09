#include "cycle_enum/core/histogram.hpp"

#include <limits>
#include <stdexcept>

#include <gtest/gtest.h>

namespace {

TEST(HistogramTest, IncrementsAndReadsCounts) {
  cycle_enum::CycleHistogram histogram;

  histogram.increment(3);
  histogram.increment(3, 4);
  histogram.increment(5, 2);

  EXPECT_EQ(histogram.count(2), 0U);
  EXPECT_EQ(histogram.count(3), 5U);
  EXPECT_EQ(histogram.count(5), 2U);
  EXPECT_EQ(histogram.total(), 7U);
}

TEST(HistogramTest, MergesWithDeterministicOrdering) {
  cycle_enum::CycleHistogram left;
  left.increment(4, 3);
  left.increment(2, 1);

  cycle_enum::CycleHistogram right;
  right.increment(3, 2);
  right.increment(4, 5);

  left.merge(right);

  const std::string expected =
      "# cycle_size, num_of_cycles\n"
      "2, 1\n"
      "3, 2\n"
      "4, 8\n"
      "Total, 11\n";
  EXPECT_EQ(left.to_csv(), expected);
}

TEST(HistogramTest, SupportsEqualityAndClear) {
  cycle_enum::CycleHistogram first;
  cycle_enum::CycleHistogram second;

  first.increment(2, 9);
  second.increment(2, 9);

  EXPECT_EQ(first, second);

  second.clear();
  EXPECT_TRUE(second.empty());
  EXPECT_NE(first, second);
}

TEST(HistogramTest, RejectsInvalidCycleLength) {
  cycle_enum::CycleHistogram histogram;

  EXPECT_THROW(histogram.increment(0), std::invalid_argument);
  EXPECT_THROW(histogram.increment(1), std::invalid_argument);
}

TEST(HistogramTest, DetectsCountOverflow) {
  cycle_enum::CycleHistogram histogram;

  histogram.increment(2, std::numeric_limits<cycle_enum::CycleHistogram::Count>::max());

  EXPECT_THROW(histogram.increment(2), std::overflow_error);
}

TEST(HistogramTest, AllowsZeroIncrementWithoutCreatingEntry) {
  cycle_enum::CycleHistogram histogram;

  histogram.increment(2, 0);

  EXPECT_TRUE(histogram.empty());
  EXPECT_EQ(histogram.to_csv(false), "# cycle_size, num_of_cycles\n");
}

}  // namespace

