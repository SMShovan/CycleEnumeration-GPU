#include "cycle_enum/core/result.hpp"

namespace cycle_enum {

std::string_view to_string(const RunStatus status) noexcept {
  switch (status) {
    case RunStatus::Success:
      return "success";
    case RunStatus::InvalidOptions:
      return "invalid-options";
    case RunStatus::Unsupported:
      return "unsupported";
    case RunStatus::RuntimeError:
      return "runtime-error";
    case RunStatus::Skipped:
      return "skipped";
  }
  return "unknown";
}

}  // namespace cycle_enum

