#pragma once

#include "cycle_enum/core/graph_view.hpp"
#include "cycle_enum/core/histogram.hpp"

#include <cstddef>

/**
 * @file cuda_johnson.hpp
 * @brief Naive CUDA Johnson-style directed-cycle counters.
 */

namespace cycle_enum::cuda {

/**
 * @brief Count static simple cycles on one CUDA device.
 *
 * This first CUDA baseline assigns one root vertex to one GPU thread and uses a
 * bounded per-thread DFS path stack. `max_cycle_length` is required because the
 * device stack must be allocated before kernel launch. Later optimized kernels
 * can replace this with more scalable work queues and stack management.
 *
 * @param graph CSR/CSC graph view.
 * @param device_id CUDA device ordinal.
 * @param max_cycle_length Maximum cycle length to count; must be at least 2.
 * @return Histogram keyed by cycle length.
 *
 * @throws std::invalid_argument if `max_cycle_length` is less than 2.
 * @throws std::runtime_error if CUDA support is unavailable in this build.
 * @throws std::out_of_range if the requested CUDA device is not visible.
 */
[[nodiscard]] CycleHistogram count_simple_cycles_johnson(
    const GraphView& graph,
    int device_id,
    std::size_t max_cycle_length);

/**
 * @brief Count simple cycles constrained by a start-edge time window.
 *
 * The naive time-window CUDA baseline assigns one start-edge timestamp to one
 * GPU thread. Each thread performs a bounded DFS and uses device binary search
 * over sorted edge timestamp ranges to preserve the CPU implementation's
 * inclusive/strict boundary convention. This is intentionally correctness
 * oriented; later kernels will replace the per-thread DFS with more balanced
 * work queues and lower-contention aggregation.
 *
 * @param graph CSR/CSC graph view.
 * @param device_id CUDA device ordinal.
 * @param window_width Inclusive time-window width.
 * @param max_cycle_length Maximum cycle length to count; must be at least 2.
 * @return Histogram keyed by cycle length.
 *
 * @throws std::invalid_argument if `window_width` is negative or
 * `max_cycle_length` is less than 2.
 * @throws std::runtime_error if CUDA support is unavailable in this build.
 * @throws std::out_of_range if the requested CUDA device is not visible.
 */
[[nodiscard]] CycleHistogram count_time_window_cycles_johnson(
    const GraphView& graph,
    int device_id,
    Timestamp window_width,
    std::size_t max_cycle_length);

}  // namespace cycle_enum::cuda
