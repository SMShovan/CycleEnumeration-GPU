#include "cycle_enum/openmp/openmp_config.hpp"

#include <stdexcept>

/**
 * @file openmp_config.cpp
 * @brief Optional OpenMP runtime detection helpers.
 */

/**
 * @def CYCLE_ENUM_OPENMP_ENABLED
 * @brief Compile-time flag set by CMake when OpenMP support is linked.
 */
#ifndef CYCLE_ENUM_OPENMP_ENABLED
#define CYCLE_ENUM_OPENMP_ENABLED 0
#endif

#if CYCLE_ENUM_OPENMP_ENABLED
#include <omp.h>
#endif

namespace cycle_enum::openmp {

OpenMPConfig config() noexcept {
#if CYCLE_ENUM_OPENMP_ENABLED
  return OpenMPConfig{true, omp_get_max_threads()};
#else
  return OpenMPConfig{false, 1};
#endif
}

bool available() noexcept {
  return config().available;
}

int max_threads() noexcept {
  return config().max_threads;
}

int resolve_thread_count(const int requested_threads) {
  if (requested_threads <= 0) {
    throw std::invalid_argument("OpenMP thread count must be positive");
  }

  if (!available() && requested_threads > 1) {
    throw std::runtime_error(
        "OpenMP support is not available in this build");
  }

  return requested_threads;
}

std::string availability_string() {
  const OpenMPConfig current = config();
  if (!current.available) {
    return "unavailable";
  }
  return "available,max_threads=" + std::to_string(current.max_threads);
}

}  // namespace cycle_enum::openmp
