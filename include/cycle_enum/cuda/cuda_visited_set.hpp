#pragma once

#include <cstddef>
#include <cstdint>

/**
 * @file cuda_visited_set.hpp
 * @brief Visited-set representations for CUDA depth-first cycle search.
 *
 * The primitives below operate on a caller-provided word array so device code
 * can place the words in registers, shared memory, or local memory. They carry
 * the `__host__ __device__` qualifiers under nvcc and compile as ordinary host
 * functions otherwise, which keeps them unit testable without a CUDA device.
 */

#if defined(__CUDACC__)
#define CYCLE_ENUM_HD __host__ __device__
#else
#define CYCLE_ENUM_HD
#endif

namespace cycle_enum::cuda {

/// Word type backing the bitset visited representation.
using BitsetWord = std::uint32_t;

/// Number of bits stored per bitset word.
inline constexpr std::size_t kBitsPerWord = 32;

/**
 * @brief How a kernel should track visited vertices for a search.
 */
enum class VisitedMode {
  Sparse, ///< Linear scan of the current path; best for short paths.
  Bitset, ///< One bit per vertex; best when the bitset fits in fast memory.
};

/**
 * @brief Number of words needed to store `bit_count` bits.
 */
CYCLE_ENUM_HD inline std::size_t bitset_word_count(const std::size_t bit_count) {
  return (bit_count + kBitsPerWord - 1) / kBitsPerWord;
}

/**
 * @brief Mark a vertex visited.
 */
CYCLE_ENUM_HD inline void bitset_set(BitsetWord* words, const std::size_t index) {
  words[index / kBitsPerWord] |=
      (static_cast<BitsetWord>(1) << (index % kBitsPerWord));
}

/**
 * @brief Mark a vertex unvisited.
 */
CYCLE_ENUM_HD inline void bitset_clear(BitsetWord* words,
                                       const std::size_t index) {
  words[index / kBitsPerWord] &=
      ~(static_cast<BitsetWord>(1) << (index % kBitsPerWord));
}

/**
 * @brief Test whether a vertex is visited.
 */
CYCLE_ENUM_HD inline bool bitset_test(const BitsetWord* words,
                                      const std::size_t index) {
  return ((words[index / kBitsPerWord] >> (index % kBitsPerWord)) &
          static_cast<BitsetWord>(1)) != 0;
}

/**
 * @brief Clear all bits backing `bit_count` vertices.
 */
CYCLE_ENUM_HD inline void bitset_clear_all(BitsetWord* words,
                                           const std::size_t bit_count) {
  const std::size_t words_needed = bitset_word_count(bit_count);
  for (std::size_t word = 0; word < words_needed; ++word) {
    words[word] = 0;
  }
}

/**
 * @brief Choose a visited representation for a graph.
 *
 * A bitset is used when the per-search word array stays at or below
 * `max_bitset_words`, so it fits in fast on-chip memory; otherwise the search
 * falls back to the sparse path scan. The threshold is a tuning parameter.
 *
 * @param vertex_count Number of vertices in the graph.
 * @param max_bitset_words Largest acceptable bitset word count.
 * @return The visited mode to use.
 */
CYCLE_ENUM_HD inline VisitedMode choose_visited_mode(
    const std::size_t vertex_count,
    const std::size_t max_bitset_words) {
  return bitset_word_count(vertex_count) <= max_bitset_words
             ? VisitedMode::Bitset
             : VisitedMode::Sparse;
}

}  // namespace cycle_enum::cuda
