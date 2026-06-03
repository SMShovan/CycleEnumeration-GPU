#include "cycle_enum/dynamic/edge_change.hpp"

#include <vector>

#include <gtest/gtest.h>

namespace {

TEST(EdgeChangeTest, OrdersBySourceThenTarget) {
  const cycle_enum::dynamic::EdgeChange a{0, 5};
  const cycle_enum::dynamic::EdgeChange b{0, 7};
  const cycle_enum::dynamic::EdgeChange c{1, 0};

  EXPECT_TRUE(a < b);
  EXPECT_TRUE(b < c);
  EXPECT_FALSE(b < a);
  EXPECT_TRUE(a == (cycle_enum::dynamic::EdgeChange{0, 5}));
  EXPECT_TRUE(a != b);
}

TEST(EdgeChangeTest, SortAndDedupAssignsOwnershipOrder) {
  std::vector<cycle_enum::dynamic::EdgeChange> changes = {
      {2, 1}, {0, 3}, {2, 1}, {0, 1},
  };
  cycle_enum::dynamic::sort_and_dedup(changes);

  ASSERT_EQ(changes.size(), 3u); // one duplicate removed
  EXPECT_EQ(changes[0], (cycle_enum::dynamic::EdgeChange{0, 1}));
  EXPECT_EQ(changes[1], (cycle_enum::dynamic::EdgeChange{0, 3}));
  EXPECT_EQ(changes[2], (cycle_enum::dynamic::EdgeChange{2, 1}));
}

TEST(EdgeChangeTest, NormalizeBothListsAndReportsTotal) {
  cycle_enum::dynamic::EdgeBatch batch;
  batch.deletions = {{1, 0}, {0, 1}, {1, 0}};
  batch.insertions = {{3, 2}, {2, 3}};
  cycle_enum::dynamic::normalize(batch);

  EXPECT_EQ(batch.deletions.size(), 2u);
  EXPECT_EQ(batch.insertions.size(), 2u);
  EXPECT_EQ(batch.deletions.front(), (cycle_enum::dynamic::EdgeChange{0, 1}));
  EXPECT_EQ(batch.total_changes(), 4u);
}

}  // namespace
