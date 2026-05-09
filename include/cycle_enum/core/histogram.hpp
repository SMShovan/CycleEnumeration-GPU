#pragma once

#include <cstdint>
#include <map>
#include <string>

/**
 * @file histogram.hpp
 * @brief Deterministic cycle-count histogram utilities.
 */

namespace cycle_enum {

/**
 * @brief Counts cycles by cycle length.
 *
 * The map-based representation keeps output deterministic, which matters for
 * tests, CLI output, benchmark logs, and comparisons with the TBB baseline.
 * Counts use 64-bit unsigned integers because cycle enumeration can produce a
 * very large number of matches even for moderate graph sizes.
 */
class CycleHistogram {
 public:
  using Length = std::uint32_t; ///< Cycle length type.
  using Count = std::uint64_t;  ///< Cycle count type.
  using Entries = std::map<Length, Count>; ///< Ordered length/count entries.

  /**
   * @brief Increment the count for a cycle length.
   *
   * @throws std::invalid_argument if `length` is less than 2.
   * @throws std::overflow_error if adding `count` would overflow.
   */
  void increment(Length length, Count count = 1);

  /**
   * @brief Merge another histogram into this histogram.
   *
   * @throws std::overflow_error if any merged bin would overflow.
   */
  void merge(const CycleHistogram& other);

  /**
   * @brief Return the count for `length`, or zero if the length is absent.
   */
  [[nodiscard]] Count count(Length length) const noexcept;

  /**
   * @brief Return the sum of all cycle counts.
   *
   * @throws std::overflow_error if the total cannot fit in Count.
   */
  [[nodiscard]] Count total() const;

  /**
   * @brief Return true when no cycle counts have been recorded.
   */
  [[nodiscard]] bool empty() const noexcept;

  /**
   * @brief Remove all entries.
   */
  void clear() noexcept;

  /**
   * @brief Return ordered histogram entries.
   */
  [[nodiscard]] const Entries& entries() const noexcept;

  /**
   * @brief Format the histogram using the baseline-compatible CSV style.
   */
  [[nodiscard]] std::string to_csv(bool include_total = true) const;

  /**
   * @brief Compare two histograms by their ordered entries.
   */
  friend bool operator==(const CycleHistogram& lhs,
                         const CycleHistogram& rhs) noexcept {
    return lhs.entries_ == rhs.entries_;
  }

  /**
   * @brief Return true when two histograms contain different entries.
   */
  friend bool operator!=(const CycleHistogram& lhs,
                         const CycleHistogram& rhs) noexcept {
    return !(lhs == rhs);
  }

 private:
  Entries entries_;
};

}  // namespace cycle_enum
