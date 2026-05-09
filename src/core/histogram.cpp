#include "cycle_enum/core/histogram.hpp"

#include <limits>
#include <sstream>
#include <stdexcept>

namespace cycle_enum {

namespace {

void ensure_valid_length(const CycleHistogram::Length length) {
  if (length < 2) {
    throw std::invalid_argument("cycle length must be at least 2");
  }
}

void ensure_addition_safe(const CycleHistogram::Count lhs,
                          const CycleHistogram::Count rhs) {
  if (rhs > std::numeric_limits<CycleHistogram::Count>::max() - lhs) {
    throw std::overflow_error("cycle histogram count overflow");
  }
}

}  // namespace

void CycleHistogram::increment(const Length length, const Count count) {
  ensure_valid_length(length);
  if (count == 0) {
    return;
  }

  Count& current = entries_[length];
  ensure_addition_safe(current, count);
  current += count;
}

void CycleHistogram::merge(const CycleHistogram& other) {
  for (const auto& [length, count_value] : other.entries_) {
    increment(length, count_value);
  }
}

CycleHistogram::Count CycleHistogram::count(const Length length) const noexcept {
  const auto found = entries_.find(length);
  return found == entries_.end() ? 0 : found->second;
}

CycleHistogram::Count CycleHistogram::total() const {
  Count sum = 0;
  for (const auto& [unused_length, count_value] : entries_) {
    (void)unused_length;
    ensure_addition_safe(sum, count_value);
    sum += count_value;
  }
  return sum;
}

bool CycleHistogram::empty() const noexcept {
  return entries_.empty();
}

void CycleHistogram::clear() noexcept {
  entries_.clear();
}

const CycleHistogram::Entries& CycleHistogram::entries() const noexcept {
  return entries_;
}

std::string CycleHistogram::to_csv(const bool include_total) const {
  std::ostringstream out;
  out << "# cycle_size, num_of_cycles\n";
  for (const auto& [length, count_value] : entries_) {
    out << length << ", " << count_value << '\n';
  }
  if (include_total) {
    out << "Total, " << total() << '\n';
  }
  return out.str();
}

}  // namespace cycle_enum

