#include "cycle_enum/cuda/cuda_profiling.hpp"

#include <gtest/gtest.h>

namespace {

// Without NVTX the scoped range must compile and be a no-op; with NVTX it must
// still construct and destruct cleanly. Either way this guards against the
// instrumentation breaking the non-CUDA build.
TEST(CudaProfilingTest, ScopedRangeConstructsAndDestructs) {
  {
    const cycle_enum::cuda::ScopedRange outer("test_outer");
    {
      const cycle_enum::cuda::ScopedRange inner("test_inner");
      SUCCEED();
    }
  }
  SUCCEED();
}

}  // namespace
