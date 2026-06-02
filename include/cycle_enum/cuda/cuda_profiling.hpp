#pragma once

/**
 * @file cuda_profiling.hpp
 * @brief NVTX scoped ranges for Nsight-oriented CUDA profiling.
 *
 * The ranges are no-ops unless the build defines `CYCLE_ENUM_WITH_NVTX`, which
 * CMake sets when the NVTX library is available on a CUDA build. This keeps the
 * instrumentation in the source on every platform while only pulling in NVTX on
 * the profiling build, so non-CUDA hosts still compile and test the code.
 */

#if defined(CYCLE_ENUM_WITH_NVTX)
#include <nvtx3/nvToolsExt.h>
#endif

namespace cycle_enum::cuda {

/**
 * @brief RAII NVTX range that brackets a phase of work for Nsight Systems.
 *
 * Construct one at the start of a phase (transfer, preprocessing, kernel,
 * reduction); the range closes when it goes out of scope. Without NVTX the
 * constructor and destructor do nothing.
 */
class ScopedRange {
 public:
  explicit ScopedRange([[maybe_unused]] const char* name) noexcept {
#if defined(CYCLE_ENUM_WITH_NVTX)
    nvtxRangePushA(name);
#endif
  }

  ~ScopedRange() {
#if defined(CYCLE_ENUM_WITH_NVTX)
    nvtxRangePop();
#endif
  }

  ScopedRange(const ScopedRange&) = delete;
  ScopedRange& operator=(const ScopedRange&) = delete;
  ScopedRange(ScopedRange&&) = delete;
  ScopedRange& operator=(ScopedRange&&) = delete;
};

}  // namespace cycle_enum::cuda
