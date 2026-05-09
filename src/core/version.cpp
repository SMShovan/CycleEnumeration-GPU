#include "cycle_enum/core/version.hpp"

#include <sstream>

namespace cycle_enum {

Version version() noexcept {
  return Version{
      CYCLE_ENUM_VERSION_MAJOR,
      CYCLE_ENUM_VERSION_MINOR,
      CYCLE_ENUM_VERSION_PATCH,
  };
}

std::string version_string() {
  const Version current = version();
  std::ostringstream out;
  out << current.major << '.' << current.minor << '.' << current.patch;
  return out.str();
}

}  // namespace cycle_enum

