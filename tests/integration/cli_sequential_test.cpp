#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include <sys/wait.h>

#include <gtest/gtest.h>

namespace {

std::string shell_quote(const std::string_view value) {
  std::string quoted = "'";
  for (const char character : value) {
    if (character == '\'') {
      quoted += "'\\''";
    } else {
      quoted += character;
    }
  }
  quoted += "'";
  return quoted;
}

std::filesystem::path unique_output_path() {
  const auto stamp = std::chrono::steady_clock::now()
                         .time_since_epoch()
                         .count();
  return std::filesystem::temp_directory_path() /
         ("cycle_enum_cli_output_" + std::to_string(stamp) + ".txt");
}

int exit_code(const int status) {
  if (WIFEXITED(status)) {
    return WEXITSTATUS(status);
  }
  return status;
}

std::string read_file(const std::filesystem::path& path) {
  std::ifstream input(path);
  std::ostringstream buffer;
  buffer << input.rdbuf();
  return buffer.str();
}

std::string run_cli(const std::vector<std::string>& args, int& status) {
  const std::filesystem::path output_path = unique_output_path();
  std::string command = shell_quote(CYCLE_ENUM_CLI_PATH);
  for (const std::string& arg : args) {
    command += ' ';
    command += shell_quote(arg);
  }
  command += " > ";
  command += shell_quote(output_path.string());
  command += " 2>&1";

  status = exit_code(std::system(command.c_str()));
  const std::string output = read_file(output_path);
  std::filesystem::remove(output_path);
  return output;
}

std::string data_path(const std::string& filename) {
  return (std::filesystem::path(CYCLE_ENUM_TEST_DATA_DIR) / filename).string();
}

TEST(CliSequentialTest, RunsJohnsonTimeWindowMode) {
  int status = -1;
  const std::string output = run_cli(
      {
          "--input",
          data_path("reference_sample.txt"),
          "--algorithm",
          "johnson",
          "--mode",
          "simple-time-window",
          "--time-window",
          "3600",
      },
      status);

  EXPECT_EQ(status, 0);
  EXPECT_EQ(output, "# cycle_size, num_of_cycles\n"
                    "2, 1\n"
                    "3, 1\n"
                    "4, 2\n"
                    "5, 1\n"
                    "Total, 5\n");
}

TEST(CliSequentialTest, RunsReadTarjanSimpleMode) {
  int status = -1;
  const std::string output = run_cli(
      {
          "--input",
          data_path("sample_temporal.txt"),
          "--algorithm",
          "read-tarjan",
          "--mode",
          "simple",
      },
      status);

  EXPECT_EQ(status, 0);
  EXPECT_EQ(output, "# cycle_size, num_of_cycles\n"
                    "2, 1\n"
                    "Total, 1\n");
}

TEST(CliSequentialTest, RejectsMissingTimeWindow) {
  int status = -1;
  const std::string output = run_cli(
      {
          "--input",
          data_path("reference_sample.txt"),
          "--mode",
          "simple-time-window",
      },
      status);

  EXPECT_NE(status, 0);
  EXPECT_NE(output.find("time_window is required"), std::string::npos);
}

}  // namespace
