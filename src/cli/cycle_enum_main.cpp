#include "cycle_enum/core/graph.hpp"
#include "cycle_enum/core/graph_view.hpp"
#include "cycle_enum/core/options.hpp"
#include "cycle_enum/core/version.hpp"
#include "cycle_enum/cuda/cuda_johnson.hpp"
#include "cycle_enum/openmp/openmp_johnson.hpp"
#include "cycle_enum/openmp/openmp_read_tarjan.hpp"
#include "cycle_enum/openmp/openmp_temporal_johnson.hpp"
#include "cycle_enum/openmp/openmp_temporal_read_tarjan.hpp"
#include "cycle_enum/sequential/bruteforce.hpp"
#include "cycle_enum/sequential/johnson.hpp"
#include "cycle_enum/sequential/read_tarjan.hpp"
#include "cycle_enum/sequential/temporal_johnson.hpp"
#include "cycle_enum/sequential/temporal_read_tarjan.hpp"

#include <charconv>
#include <exception>
#include <filesystem>
#include <iostream>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

/**
 * @file cycle_enum_main.cpp
 * @brief Command-line entry point for CPU cycle counting modes.
 */

namespace {

struct CliConfig {
  std::filesystem::path input_path;
  cycle_enum::CycleEnumerationOptions options =
      cycle_enum::default_options();
  bool show_help = false;
  bool show_version = false;
};

void print_usage(std::ostream& out) {
  out << "Usage: cycle-enum --input <path> [options]\n\n"
      << "Options:\n"
      << "  --algorithm <johnson|read-tarjan|brute-force>\n"
      << "  --backend <sequential|openmp>\n"
      << "  --mode <simple|simple-time-window|temporal>\n"
      << "  --time-window <positive integer>\n"
      << "  --max-cycle-length <integer >= 2>\n"
      << "  --openmp-threads <positive integer>\n"
      << "  --cuda-device <non-negative integer>\n"
      << "  --help\n"
      << "  --version\n";
}

[[nodiscard]] bool is_option(const std::string_view value) noexcept {
  return value.rfind("--", 0) == 0;
}

[[nodiscard]] std::optional<std::string_view> next_value(
    const std::vector<std::string_view>& args,
    std::size_t& index,
    const std::string_view option,
    std::ostream& err) {
  if (index + 1 >= args.size() || is_option(args[index + 1])) {
    err << option << " requires a value\n";
    return std::nullopt;
  }
  ++index;
  return args[index];
}

template <typename Integer>
[[nodiscard]] std::optional<Integer> parse_integer(
    const std::string_view value,
    const std::string_view option,
    std::ostream& err) {
  Integer parsed{};
  const char* const begin = value.data();
  const char* const end = value.data() + value.size();
  const auto [position, error] = std::from_chars(begin, end, parsed);
  if (error != std::errc{} || position != end) {
    err << option << " expects an integer value\n";
    return std::nullopt;
  }
  return parsed;
}

[[nodiscard]] std::optional<cycle_enum::AlgorithmFamily> parse_algorithm(
    const std::string_view value) noexcept {
  if (value == "johnson") {
    return cycle_enum::AlgorithmFamily::Johnson;
  }
  if (value == "read-tarjan" || value == "read_tarjan" ||
      value == "readtarjan") {
    return cycle_enum::AlgorithmFamily::ReadTarjan;
  }
  if (value == "brute-force" || value == "bruteforce" ||
      value == "brute_force") {
    return cycle_enum::AlgorithmFamily::BruteForce;
  }
  return std::nullopt;
}

[[nodiscard]] std::optional<cycle_enum::ExecutionPolicy> parse_execution(
    const std::string_view value) noexcept {
  if (value == "sequential" || value == "seq" || value == "cpu") {
    return cycle_enum::ExecutionPolicy::Sequential;
  }
  if (value == "openmp" || value == "omp") {
    return cycle_enum::ExecutionPolicy::OpenMP;
  }
  if (value == "cuda" || value == "gpu") {
    return cycle_enum::ExecutionPolicy::Cuda;
  }
  return std::nullopt;
}

[[nodiscard]] std::optional<cycle_enum::CycleMode> parse_mode(
    const std::string_view value) noexcept {
  if (value == "simple") {
    return cycle_enum::CycleMode::Simple;
  }
  if (value == "simple-time-window" || value == "simple_time_window" ||
      value == "time-window" || value == "window") {
    return cycle_enum::CycleMode::SimpleTimeWindow;
  }
  if (value == "temporal") {
    return cycle_enum::CycleMode::Temporal;
  }
  return std::nullopt;
}

[[nodiscard]] std::vector<std::string> validate_cli_supported(
    const cycle_enum::CycleEnumerationOptions& options) {
  std::vector<std::string> errors;

  if (options.execution == cycle_enum::ExecutionPolicy::Cuda) {
    if (options.algorithm != cycle_enum::AlgorithmFamily::Johnson) {
      errors.emplace_back("cuda backend currently supports only johnson");
    }
    if (options.mode != cycle_enum::CycleMode::Simple) {
      errors.emplace_back("cuda backend currently supports only simple mode");
    }
    if (!options.max_cycle_length.has_value()) {
      errors.emplace_back(
          "cuda backend requires --max-cycle-length for bounded device stacks");
    }
  }

  if (options.execution == cycle_enum::ExecutionPolicy::OpenMP) {
    if (options.algorithm == cycle_enum::AlgorithmFamily::BruteForce) {
      errors.emplace_back("brute-force is only available with sequential backend");
    }
    if (options.mode == cycle_enum::CycleMode::SimpleTimeWindow) {
      errors.emplace_back(
          "openmp backend does not implement simple-time-window mode yet");
    }
  }

  return errors;
}

[[nodiscard]] std::optional<CliConfig> parse_args(int argc,
                                                  char** argv,
                                                  std::ostream& err) {
  CliConfig config;
  config.options.execution = cycle_enum::ExecutionPolicy::Sequential;

  std::vector<std::string_view> args;
  args.reserve(static_cast<std::size_t>(argc));
  for (int i = 0; i < argc; ++i) {
    args.emplace_back(argv[i]);
  }

  for (std::size_t index = 1; index < args.size(); ++index) {
    const std::string_view option = args[index];

    if (option == "--help" || option == "-h") {
      config.show_help = true;
      return config;
    }
    if (option == "--version") {
      config.show_version = true;
      return config;
    }

    if (option == "--input" || option == "-i") {
      const auto value = next_value(args, index, option, err);
      if (!value.has_value()) {
        return std::nullopt;
      }
      config.input_path = std::filesystem::path(std::string(*value));
      continue;
    }

    if (option == "--algorithm" || option == "-a") {
      const auto value = next_value(args, index, option, err);
      if (!value.has_value()) {
        return std::nullopt;
      }
      const auto algorithm = parse_algorithm(*value);
      if (!algorithm.has_value()) {
        err << "unknown algorithm: " << *value << '\n';
        return std::nullopt;
      }
      config.options.algorithm = *algorithm;
      continue;
    }

    if (option == "--backend" || option == "--execution" ||
        option == "--policy") {
      const auto value = next_value(args, index, option, err);
      if (!value.has_value()) {
        return std::nullopt;
      }
      const auto execution = parse_execution(*value);
      if (!execution.has_value()) {
        err << "unknown backend: " << *value << '\n';
        return std::nullopt;
      }
      config.options.execution = *execution;
      continue;
    }

    if (option == "--mode" || option == "-m") {
      const auto value = next_value(args, index, option, err);
      if (!value.has_value()) {
        return std::nullopt;
      }
      const auto mode = parse_mode(*value);
      if (!mode.has_value()) {
        err << "unknown mode: " << *value << '\n';
        return std::nullopt;
      }
      config.options.mode = *mode;
      continue;
    }

    if (option == "--openmp-threads" || option == "--threads") {
      const auto value = next_value(args, index, option, err);
      if (!value.has_value()) {
        return std::nullopt;
      }
      const auto parsed = parse_integer<int>(*value, option, err);
      if (!parsed.has_value()) {
        return std::nullopt;
      }
      config.options.openmp_threads = *parsed;
      continue;
    }

    if (option == "--cuda-device" || option == "--device") {
      const auto value = next_value(args, index, option, err);
      if (!value.has_value()) {
        return std::nullopt;
      }
      const auto parsed = parse_integer<int>(*value, option, err);
      if (!parsed.has_value()) {
        return std::nullopt;
      }
      config.options.cuda_device_id = *parsed;
      continue;
    }

    if (option == "--time-window" || option == "--window") {
      const auto value = next_value(args, index, option, err);
      if (!value.has_value()) {
        return std::nullopt;
      }
      const auto parsed =
          parse_integer<cycle_enum::Timestamp>(*value, option, err);
      if (!parsed.has_value()) {
        return std::nullopt;
      }
      config.options.time_window = *parsed;
      continue;
    }

    if (option == "--max-cycle-length" || option == "--max-length") {
      const auto value = next_value(args, index, option, err);
      if (!value.has_value()) {
        return std::nullopt;
      }
      const auto parsed = parse_integer<std::size_t>(*value, option, err);
      if (!parsed.has_value()) {
        return std::nullopt;
      }
      config.options.max_cycle_length = *parsed;
      continue;
    }

    err << "unknown option: " << option << '\n';
    return std::nullopt;
  }

  if (config.input_path.empty()) {
    err << "--input is required\n";
    return std::nullopt;
  }

  const std::vector<std::string> errors =
      cycle_enum::validate_options(config.options);
  if (!errors.empty()) {
    for (const std::string& error : errors) {
      err << error << '\n';
    }
    return std::nullopt;
  }

  const std::vector<std::string> cli_errors =
      validate_cli_supported(config.options);
  if (!cli_errors.empty()) {
    for (const std::string& error : cli_errors) {
      err << error << '\n';
    }
    return std::nullopt;
  }

  return config;
}

[[nodiscard]] cycle_enum::CycleHistogram run_sequential(
    const cycle_enum::GraphView& graph,
    const cycle_enum::CycleEnumerationOptions& options) {
  switch (options.mode) {
    case cycle_enum::CycleMode::Simple:
      switch (options.algorithm) {
        case cycle_enum::AlgorithmFamily::Johnson:
          return cycle_enum::sequential::count_simple_cycles_johnson(
              graph, options.max_cycle_length);
        case cycle_enum::AlgorithmFamily::ReadTarjan:
          return cycle_enum::sequential::count_simple_cycles_read_tarjan(
              graph, options.max_cycle_length);
        case cycle_enum::AlgorithmFamily::BruteForce:
          return cycle_enum::sequential::count_simple_cycles_bruteforce(
              graph, options.max_cycle_length);
      }
      break;

    case cycle_enum::CycleMode::SimpleTimeWindow:
      switch (options.algorithm) {
        case cycle_enum::AlgorithmFamily::Johnson:
          return cycle_enum::sequential::count_time_window_cycles_johnson(
              graph, *options.time_window, options.max_cycle_length);
        case cycle_enum::AlgorithmFamily::ReadTarjan:
          return cycle_enum::sequential::count_time_window_cycles_read_tarjan(
              graph, *options.time_window, options.max_cycle_length);
        case cycle_enum::AlgorithmFamily::BruteForce:
          return cycle_enum::sequential::count_time_window_cycles_bruteforce(
              graph, *options.time_window, options.max_cycle_length);
      }
      break;

    case cycle_enum::CycleMode::Temporal:
      switch (options.algorithm) {
        case cycle_enum::AlgorithmFamily::Johnson:
          return cycle_enum::sequential::count_temporal_cycles_johnson(
              graph, *options.time_window, options.max_cycle_length);
        case cycle_enum::AlgorithmFamily::ReadTarjan:
          return cycle_enum::sequential::count_temporal_cycles_read_tarjan(
              graph, *options.time_window, options.max_cycle_length);
        case cycle_enum::AlgorithmFamily::BruteForce:
          return cycle_enum::sequential::count_temporal_cycles_bruteforce(
              graph, *options.time_window, options.max_cycle_length);
      }
      break;
  }

  throw std::logic_error("unsupported sequential cycle counting mode");
}

[[nodiscard]] cycle_enum::CycleHistogram run_openmp(
    const cycle_enum::GraphView& graph,
    const cycle_enum::CycleEnumerationOptions& options) {
  switch (options.mode) {
    case cycle_enum::CycleMode::Simple:
      switch (options.algorithm) {
        case cycle_enum::AlgorithmFamily::Johnson:
          return cycle_enum::openmp::count_simple_cycles_johnson(
              graph, options.openmp_threads, options.max_cycle_length);
        case cycle_enum::AlgorithmFamily::ReadTarjan:
          return cycle_enum::openmp::count_simple_cycles_read_tarjan(
              graph, options.openmp_threads, options.max_cycle_length);
        case cycle_enum::AlgorithmFamily::BruteForce:
          break;
      }
      break;

    case cycle_enum::CycleMode::Temporal:
      switch (options.algorithm) {
        case cycle_enum::AlgorithmFamily::Johnson:
          return cycle_enum::openmp::count_temporal_cycles_johnson(
              graph, *options.time_window, options.openmp_threads,
              options.max_cycle_length);
        case cycle_enum::AlgorithmFamily::ReadTarjan:
          return cycle_enum::openmp::count_temporal_cycles_read_tarjan(
              graph, *options.time_window, options.openmp_threads,
              options.max_cycle_length);
        case cycle_enum::AlgorithmFamily::BruteForce:
          break;
      }
      break;

    case cycle_enum::CycleMode::SimpleTimeWindow:
      break;
  }

  throw std::logic_error("unsupported OpenMP cycle counting mode");
}

[[nodiscard]] cycle_enum::CycleHistogram run_cuda(
    const cycle_enum::GraphView& graph,
    const cycle_enum::CycleEnumerationOptions& options) {
  if (options.mode == cycle_enum::CycleMode::Simple &&
      options.algorithm == cycle_enum::AlgorithmFamily::Johnson &&
      options.max_cycle_length.has_value()) {
    return cycle_enum::cuda::count_simple_cycles_johnson(
        graph, options.cuda_device_id, *options.max_cycle_length);
  }

  throw std::logic_error("unsupported CUDA cycle counting mode");
}

[[nodiscard]] cycle_enum::CycleHistogram run_backend(
    const cycle_enum::GraphView& graph,
    const cycle_enum::CycleEnumerationOptions& options) {
  switch (options.execution) {
    case cycle_enum::ExecutionPolicy::Sequential:
      return run_sequential(graph, options);
    case cycle_enum::ExecutionPolicy::OpenMP:
      return run_openmp(graph, options);
    case cycle_enum::ExecutionPolicy::Cuda:
      return run_cuda(graph, options);
  }

  throw std::logic_error("unsupported execution backend");
}

}  // namespace

/**
 * @brief Parse CLI arguments, run the requested CPU counter, and print the
 * cycle histogram.
 */
int main(int argc, char** argv) {
  const std::optional<CliConfig> config = parse_args(argc, argv, std::cerr);
  if (!config.has_value()) {
    print_usage(std::cerr);
    return 1;
  }

  if (config->show_help) {
    print_usage(std::cout);
    return 0;
  }

  if (config->show_version) {
    std::cout << cycle_enum::version_string() << '\n';
    return 0;
  }

  try {
    const cycle_enum::TemporalGraph parsed =
        cycle_enum::read_temporal_graph(config->input_path);
    const cycle_enum::GraphView view = cycle_enum::build_graph_view(parsed);
    const cycle_enum::CycleHistogram histogram =
        run_backend(view, config->options);
    std::cout << histogram.to_csv();
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }

  return 0;
}
