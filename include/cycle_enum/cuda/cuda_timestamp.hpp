#pragma once

#include "cycle_enum/core/graph.hpp"

#include <cstdint>
#include <limits>

/**
 * @file cuda_timestamp.hpp
 * @brief Host/device timestamp search helpers for CUDA kernels.
 */

#if defined(__CUDACC__)
#define CYCLE_ENUM_CUDA_HOST_DEVICE __host__ __device__
#else
#define CYCLE_ENUM_CUDA_HOST_DEVICE
#endif

namespace cycle_enum::cuda {

/**
 * @brief Offset range into the flattened device timestamp array.
 *
 * CUDA kernels use integer offsets rather than STL iterators. The range is
 * half-open: `begin` is inclusive and `end` is exclusive.
 */
struct CudaTimestampRange {
  std::uint64_t begin = 0; ///< Inclusive begin offset.
  std::uint64_t end = 0;   ///< Exclusive end offset.

  /**
   * @brief Return true when the range contains no timestamps.
   */
  CYCLE_ENUM_CUDA_HOST_DEVICE constexpr bool empty()
      const noexcept {
    return begin >= end;
  }

  /**
   * @brief Return the number of timestamps in the range.
   */
  CYCLE_ENUM_CUDA_HOST_DEVICE constexpr std::uint64_t size()
      const noexcept {
    return empty() ? 0 : end - begin;
  }
};

/**
 * @brief Lower-bound policy for a device time-window lookup.
 */
enum class CudaTimestampStartPolicy {
  Inclusive, ///< Include timestamps equal to the window start.
  AfterStart ///< Require timestamps to be strictly greater than the start.
};

/**
 * @brief Compute a timestamp window end without throwing on device.
 *
 * Host-side option validation rejects negative windows and overflow before
 * kernels are launched. Device helpers still return `false` for invalid input
 * so accidental bad launch parameters produce empty ranges instead of wrapping.
 */
CYCLE_ENUM_CUDA_HOST_DEVICE constexpr bool checked_window_end(
    const Timestamp window_start,
    const Timestamp window_width,
    Timestamp& window_end) noexcept {
  if (window_width < 0) {
    return false;
  }
  if (window_start > std::numeric_limits<Timestamp>::max() - window_width) {
    return false;
  }
  window_end = window_start + window_width;
  return true;
}

/**
 * @brief Return the first offset whose value is not less than `target`.
 */
template <typename Value>
CYCLE_ENUM_CUDA_HOST_DEVICE constexpr std::uint64_t
timestamp_lower_bound(const Value* values,
                      std::uint64_t begin,
                      std::uint64_t end,
                      const Value target) noexcept {
  while (begin < end) {
    const std::uint64_t mid = begin + ((end - begin) / 2);
    if (values[mid] < target) {
      begin = mid + 1;
    } else {
      end = mid;
    }
  }
  return begin;
}

/**
 * @brief Return the first offset whose value is greater than `target`.
 */
template <typename Value>
CYCLE_ENUM_CUDA_HOST_DEVICE constexpr std::uint64_t
timestamp_upper_bound(const Value* values,
                      std::uint64_t begin,
                      std::uint64_t end,
                      const Value target) noexcept {
  while (begin < end) {
    const std::uint64_t mid = begin + ((end - begin) / 2);
    if (!(target < values[mid])) {
      begin = mid + 1;
    } else {
      end = mid;
    }
  }
  return begin;
}

/**
 * @brief Find timestamps inside `[window_start, window_start + width]`.
 *
 * The input subrange must already be sorted in nondecreasing order. Returned
 * offsets are relative to the full flattened timestamp array.
 */
CYCLE_ENUM_CUDA_HOST_DEVICE constexpr CudaTimestampRange
timestamps_in_window(const Timestamp* timestamps,
                     const std::uint64_t search_begin,
                     const std::uint64_t search_end,
                     const Timestamp window_start,
                     const Timestamp window_width,
                     const CudaTimestampStartPolicy start_policy =
                         CudaTimestampStartPolicy::Inclusive) noexcept {
  Timestamp window_end = 0;
  if (!checked_window_end(window_start, window_width, window_end) ||
      search_begin > search_end) {
    return {};
  }

  const std::uint64_t begin =
      start_policy == CudaTimestampStartPolicy::Inclusive
          ? timestamp_lower_bound(timestamps, search_begin, search_end,
                                  window_start)
          : timestamp_upper_bound(timestamps, search_begin, search_end,
                                  window_start);
  const std::uint64_t end =
      timestamp_upper_bound(timestamps, begin, search_end, window_end);
  return CudaTimestampRange{begin, end};
}

/**
 * @brief Return true when a timestamp subrange has a value in a time window.
 */
CYCLE_ENUM_CUDA_HOST_DEVICE constexpr bool has_timestamp_in_window(
    const Timestamp* timestamps,
    const std::uint64_t search_begin,
    const std::uint64_t search_end,
    const Timestamp window_start,
    const Timestamp window_width,
    const CudaTimestampStartPolicy start_policy =
        CudaTimestampStartPolicy::Inclusive) noexcept {
  return !timestamps_in_window(timestamps, search_begin, search_end,
                               window_start, window_width, start_policy)
              .empty();
}

/**
 * @brief Find temporal timestamps in `(previous_timestamp, window_end]`.
 *
 * This is the timestamp search used by temporal cycle enumeration: every next
 * edge event must be strictly later than the previous edge event while staying
 * inside the start edge's window.
 */
CYCLE_ENUM_CUDA_HOST_DEVICE constexpr CudaTimestampRange
timestamps_after(const Timestamp* timestamps,
                 const std::uint64_t search_begin,
                 const std::uint64_t search_end,
                 const Timestamp previous_timestamp,
                 const Timestamp window_end) noexcept {
  if (search_begin > search_end) {
    return {};
  }
  const std::uint64_t begin =
      timestamp_upper_bound(timestamps, search_begin, search_end,
                            previous_timestamp);
  const std::uint64_t end =
      timestamp_upper_bound(timestamps, begin, search_end, window_end);
  return CudaTimestampRange{begin, end};
}

}  // namespace cycle_enum::cuda

#undef CYCLE_ENUM_CUDA_HOST_DEVICE
