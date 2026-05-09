#include "cycle_enum/core/options.hpp"

#include <sstream>

namespace cycle_enum {

CycleEnumerationOptions default_options() noexcept {
  return CycleEnumerationOptions{};
}

bool requires_time_window(const CycleMode mode) noexcept {
  return mode == CycleMode::SimpleTimeWindow || mode == CycleMode::Temporal;
}

std::string_view to_string(const CycleMode mode) noexcept {
  switch (mode) {
    case CycleMode::Simple:
      return "simple";
    case CycleMode::SimpleTimeWindow:
      return "simple-time-window";
    case CycleMode::Temporal:
      return "temporal";
  }
  return "unknown";
}

std::string_view to_string(const AlgorithmFamily algorithm) noexcept {
  switch (algorithm) {
    case AlgorithmFamily::Johnson:
      return "johnson";
    case AlgorithmFamily::ReadTarjan:
      return "read-tarjan";
    case AlgorithmFamily::BruteForce:
      return "brute-force";
  }
  return "unknown";
}

std::string_view to_string(const ExecutionPolicy execution) noexcept {
  switch (execution) {
    case ExecutionPolicy::Sequential:
      return "sequential";
    case ExecutionPolicy::OpenMP:
      return "openmp";
    case ExecutionPolicy::Cuda:
      return "cuda";
  }
  return "unknown";
}

std::vector<std::string> validate_options(
    const CycleEnumerationOptions& options) {
  std::vector<std::string> errors;

  if (requires_time_window(options.mode)) {
    if (!options.time_window.has_value()) {
      errors.emplace_back("time_window is required for the selected cycle mode");
    } else if (*options.time_window <= 0) {
      errors.emplace_back("time_window must be positive");
    }
  }

  if (options.time_window.has_value() && *options.time_window <= 0) {
    errors.emplace_back("time_window must be positive when provided");
  }

  if (options.max_cycle_length.has_value() && *options.max_cycle_length < 2) {
    errors.emplace_back("max_cycle_length must be at least 2 when provided");
  }

  if (options.openmp_threads <= 0) {
    errors.emplace_back("openmp_threads must be positive");
  }

  if (options.cuda_device_id < 0) {
    errors.emplace_back("cuda_device_id must be non-negative");
  }

  if (options.use_cycle_union && options.mode != CycleMode::Temporal) {
    errors.emplace_back("cycle-union pruning is only valid for temporal mode");
  }

  return errors;
}

}  // namespace cycle_enum

