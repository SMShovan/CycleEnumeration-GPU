#pragma once

#include "cycle_enum/core/graph.hpp"

#include <cstddef>
#include <optional>
#include <vector>

/**
 * @file timestamp.hpp
 * @brief Host-side timestamp interval helpers.
 */

namespace cycle_enum {

/**
 * @brief Offset range into a sorted timestamp vector.
 */
struct TimestampRange {
  std::size_t begin = 0; ///< Inclusive begin offset.
  std::size_t end = 0;   ///< Exclusive end offset.

  /**
   * @brief Return true when the range contains no timestamps.
   */
  [[nodiscard]] bool empty() const noexcept;

  /**
   * @brief Return the number of timestamps in the range.
   */
  [[nodiscard]] std::size_t size() const noexcept;
};

/**
 * @brief Lower-bound policy for a time-window lookup.
 */
enum class TimestampStartPolicy {
  Inclusive, ///< Include timestamps equal to the window start.
  AfterStart ///< Require timestamps to be strictly greater than the start.
};

/**
 * @brief Find timestamps inside `[window_start, window_start + width]`.
 *
 * `timestamps` must be sorted in nondecreasing order. Duplicate timestamps are
 * preserved and included when they fall inside the requested range.
 *
 * @throws std::invalid_argument if `window_width` is negative.
 * @throws std::overflow_error if the window end overflows Timestamp.
 */
[[nodiscard]] TimestampRange timestamps_in_window(
    const std::vector<Timestamp>& timestamps,
    Timestamp window_start,
    Timestamp window_width,
    TimestampStartPolicy start_policy = TimestampStartPolicy::Inclusive);

/**
 * @brief Return whether a sorted timestamp vector has a value in a window.
 */
[[nodiscard]] bool has_timestamp_in_window(
    const std::vector<Timestamp>& timestamps,
    Timestamp window_start,
    Timestamp window_width,
    TimestampStartPolicy start_policy = TimestampStartPolicy::Inclusive);

/**
 * @brief Find temporal timestamps in `(previous_timestamp, window_end]`.
 *
 * This helper implements the strict timestamp increase used by temporal cycle
 * enumeration.
 */
[[nodiscard]] TimestampRange timestamps_after(
    const std::vector<Timestamp>& timestamps,
    Timestamp previous_timestamp,
    Timestamp window_end);

/**
 * @brief Return the first timestamp in a range, if the range is not empty.
 */
[[nodiscard]] std::optional<Timestamp> first_timestamp(
    const std::vector<Timestamp>& timestamps,
    TimestampRange range);

}  // namespace cycle_enum

