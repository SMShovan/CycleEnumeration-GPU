#include "cycle_enum/core/timestamp.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace cycle_enum {

namespace {

Timestamp checked_window_end(const Timestamp window_start,
                             const Timestamp window_width) {
  if (window_width < 0) {
    throw std::invalid_argument("time window width must be non-negative");
  }
  if (window_start > std::numeric_limits<Timestamp>::max() - window_width) {
    throw std::overflow_error("time window end overflows timestamp range");
  }
  return window_start + window_width;
}

}  // namespace

bool TimestampRange::empty() const noexcept {
  return begin >= end;
}

std::size_t TimestampRange::size() const noexcept {
  return empty() ? 0 : end - begin;
}

TimestampRange timestamps_in_window(
    const std::vector<Timestamp>& timestamps,
    const Timestamp window_start,
    const Timestamp window_width,
    const TimestampStartPolicy start_policy) {
  return timestamps_in_window(timestamps, 0, timestamps.size(), window_start,
                              window_width, start_policy);
}

TimestampRange timestamps_in_window(
    const std::vector<Timestamp>& timestamps,
    const std::size_t search_begin,
    const std::size_t search_end,
    const Timestamp window_start,
    const Timestamp window_width,
    const TimestampStartPolicy start_policy) {
  const Timestamp window_end = checked_window_end(window_start, window_width);

  if (search_begin > search_end || search_end > timestamps.size()) {
    throw std::out_of_range("timestamp search range is out of bounds");
  }

  const auto range_begin = timestamps.begin() + static_cast<std::ptrdiff_t>(search_begin);
  const auto range_end = timestamps.begin() + static_cast<std::ptrdiff_t>(search_end);
  const auto begin_it =
      start_policy == TimestampStartPolicy::Inclusive
          ? std::lower_bound(range_begin, range_end, window_start)
          : std::upper_bound(range_begin, range_end, window_start);
  const auto end_it = std::upper_bound(begin_it, range_end, window_end);

  return TimestampRange{
      static_cast<std::size_t>(std::distance(timestamps.begin(), begin_it)),
      static_cast<std::size_t>(std::distance(timestamps.begin(), end_it)),
  };
}

bool has_timestamp_in_window(const std::vector<Timestamp>& timestamps,
                             const Timestamp window_start,
                             const Timestamp window_width,
                             const TimestampStartPolicy start_policy) {
  return !timestamps_in_window(timestamps, window_start, window_width,
                               start_policy)
              .empty();
}

bool has_timestamp_in_window(const std::vector<Timestamp>& timestamps,
                             const std::size_t search_begin,
                             const std::size_t search_end,
                             const Timestamp window_start,
                             const Timestamp window_width,
                             const TimestampStartPolicy start_policy) {
  return !timestamps_in_window(timestamps, search_begin, search_end,
                               window_start, window_width, start_policy)
              .empty();
}

TimestampRange timestamps_after(const std::vector<Timestamp>& timestamps,
                                const Timestamp previous_timestamp,
                                const Timestamp window_end) {
  return timestamps_after(timestamps, 0, timestamps.size(), previous_timestamp,
                          window_end);
}

TimestampRange timestamps_after(const std::vector<Timestamp>& timestamps,
                                const std::size_t search_begin,
                                const std::size_t search_end,
                                const Timestamp previous_timestamp,
                                const Timestamp window_end) {
  if (search_begin > search_end || search_end > timestamps.size()) {
    throw std::out_of_range("timestamp search range is out of bounds");
  }

  const auto range_begin = timestamps.begin() + static_cast<std::ptrdiff_t>(search_begin);
  const auto range_end = timestamps.begin() + static_cast<std::ptrdiff_t>(search_end);
  const auto begin_it =
      std::upper_bound(range_begin, range_end, previous_timestamp);
  const auto end_it = std::upper_bound(begin_it, range_end, window_end);

  return TimestampRange{
      static_cast<std::size_t>(std::distance(timestamps.begin(), begin_it)),
      static_cast<std::size_t>(std::distance(timestamps.begin(), end_it)),
  };
}

std::optional<Timestamp> first_timestamp(
    const std::vector<Timestamp>& timestamps,
    const TimestampRange range) {
  if (range.empty() || range.begin >= timestamps.size()) {
    return std::nullopt;
  }
  return timestamps[range.begin];
}

}  // namespace cycle_enum
