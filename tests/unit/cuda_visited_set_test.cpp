#include "cycle_enum/cuda/cuda_visited_set.hpp"

#include <cstddef>
#include <vector>

#include <gtest/gtest.h>

namespace {

TEST(CudaVisitedSetTest, WordCountRoundsUp) {
  EXPECT_EQ(cycle_enum::cuda::bitset_word_count(0), 0u);
  EXPECT_EQ(cycle_enum::cuda::bitset_word_count(1), 1u);
  EXPECT_EQ(cycle_enum::cuda::bitset_word_count(32), 1u);
  EXPECT_EQ(cycle_enum::cuda::bitset_word_count(33), 2u);
  EXPECT_EQ(cycle_enum::cuda::bitset_word_count(64), 2u);
}

TEST(CudaVisitedSetTest, SetTestClearAcrossWordBoundary) {
  std::vector<cycle_enum::cuda::BitsetWord> words(
      cycle_enum::cuda::bitset_word_count(70), 0);

  // Bits that straddle the 32-bit word boundary.
  for (const std::size_t bit : {std::size_t{0}, std::size_t{31}, std::size_t{32},
                                std::size_t{63}, std::size_t{69}}) {
    EXPECT_FALSE(cycle_enum::cuda::bitset_test(words.data(), bit));
    cycle_enum::cuda::bitset_set(words.data(), bit);
    EXPECT_TRUE(cycle_enum::cuda::bitset_test(words.data(), bit));
  }

  // Setting one bit must not disturb its neighbors.
  EXPECT_FALSE(cycle_enum::cuda::bitset_test(words.data(), 30));
  EXPECT_FALSE(cycle_enum::cuda::bitset_test(words.data(), 33));

  cycle_enum::cuda::bitset_clear(words.data(), 32);
  EXPECT_FALSE(cycle_enum::cuda::bitset_test(words.data(), 32));
  EXPECT_TRUE(cycle_enum::cuda::bitset_test(words.data(), 31));
}

TEST(CudaVisitedSetTest, ClearAllResetsEveryBit) {
  std::vector<cycle_enum::cuda::BitsetWord> words(
      cycle_enum::cuda::bitset_word_count(40), 0);
  cycle_enum::cuda::bitset_set(words.data(), 5);
  cycle_enum::cuda::bitset_set(words.data(), 39);

  cycle_enum::cuda::bitset_clear_all(words.data(), 40);
  for (std::size_t bit = 0; bit < 40; ++bit) {
    EXPECT_FALSE(cycle_enum::cuda::bitset_test(words.data(), bit));
  }
}

TEST(CudaVisitedSetTest, ModeSelectionUsesWordThreshold) {
  // 64 vertices need two 32-bit words; a threshold of two keeps the bitset.
  EXPECT_EQ(cycle_enum::cuda::choose_visited_mode(64, 2),
            cycle_enum::cuda::VisitedMode::Bitset);
  // 96 vertices need three words and exceed the threshold, so fall back.
  EXPECT_EQ(cycle_enum::cuda::choose_visited_mode(96, 2),
            cycle_enum::cuda::VisitedMode::Sparse);
}

}  // namespace
