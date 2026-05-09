#pragma once

#include <string>

/**
 * @file version.hpp
 * @brief Version helpers for the CycleEnumeration-GPU libraries.
 */

namespace cycle_enum {

/**
 * @brief Semantic version components compiled into the library.
 *
 * The build system defines the numeric version macros from the top-level
 * CMake project. Keeping the values available from C++ makes CLI output,
 * benchmark logs, and later validation reports easier to reproduce.
 */
struct Version {
  int major; ///< Major version component.
  int minor; ///< Minor version component.
  int patch; ///< Patch version component.
};

/**
 * @brief Return the semantic version compiled into the core library.
 */
[[nodiscard]] Version version() noexcept;

/**
 * @brief Return the project version formatted as `major.minor.patch`.
 */
[[nodiscard]] std::string version_string();

}  // namespace cycle_enum

