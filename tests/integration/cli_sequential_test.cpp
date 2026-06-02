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

TEST(CliSequentialTest, RunsTemporalMode) {
  int status = -1;
  const std::string output = run_cli(
      {
          "--input",
          data_path("sample_temporal.txt"),
          "--algorithm",
          "johnson",
          "--mode",
          "temporal",
          "--time-window",
          "10",
      },
      status);

  EXPECT_EQ(status, 0);
  EXPECT_EQ(output, "# cycle_size, num_of_cycles\n"
                    "2, 2\n"
                    "Total, 2\n");
}

TEST(CliOpenMPTest, RunsOpenMPSimpleModeWithSingleThread) {
  int status = -1;
  const std::string output = run_cli(
      {
          "--input",
          data_path("sample_temporal.txt"),
          "--backend",
          "openmp",
          "--openmp-threads",
          "1",
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

TEST(CliOpenMPTest, RunsOpenMPTemporalModeWithSingleThread) {
  int status = -1;
  const std::string output = run_cli(
      {
          "--input",
          data_path("sample_temporal.txt"),
          "--backend",
          "openmp",
          "--openmp-threads",
          "1",
          "--algorithm",
          "johnson",
          "--mode",
          "temporal",
          "--time-window",
          "10",
      },
      status);

  EXPECT_EQ(status, 0);
  EXPECT_EQ(output, "# cycle_size, num_of_cycles\n"
                    "2, 2\n"
                    "Total, 2\n");
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

TEST(CliOpenMPTest, RejectsUnsupportedTimeWindowMode) {
  int status = -1;
  const std::string output = run_cli(
      {
          "--input",
          data_path("reference_sample.txt"),
          "--backend",
          "openmp",
          "--mode",
          "simple-time-window",
          "--time-window",
          "3600",
      },
      status);

  EXPECT_NE(status, 0);
  EXPECT_NE(output.find("openmp backend does not implement"), std::string::npos);
}

TEST(CliCudaTest, RequiresMaximumCycleLength) {
  int status = -1;
  const std::string output = run_cli(
      {
          "--input",
          data_path("sample_temporal.txt"),
          "--backend",
          "cuda",
          "--algorithm",
          "johnson",
          "--mode",
          "simple",
      },
      status);

  EXPECT_NE(status, 0);
  EXPECT_NE(output.find("cuda backend requires --max-cycle-length"),
            std::string::npos);
}

TEST(CliCudaTest, ReportsUnavailableBackendWhenNotCompiled) {
  int status = -1;
  const std::string output = run_cli(
      {
          "--input",
          data_path("sample_temporal.txt"),
          "--backend",
          "cuda",
          "--cuda-device",
          "0",
          "--algorithm",
          "johnson",
          "--mode",
          "simple",
          "--max-cycle-length",
          "4",
      },
      status);

  if (output.find("CUDA support is not compiled") == std::string::npos) {
    GTEST_SKIP() << "CUDA support is compiled in this test build";
  }

  EXPECT_NE(status, 0);
  EXPECT_NE(output.find("CUDA support is not compiled"), std::string::npos);
}

TEST(CliCudaTest, TimeWindowReportsUnavailableBackendWhenNotCompiled) {
  int status = -1;
  const std::string output = run_cli(
      {
          "--input",
          data_path("sample_temporal.txt"),
          "--backend",
          "cuda",
          "--cuda-device",
          "0",
          "--algorithm",
          "johnson",
          "--mode",
          "simple-time-window",
          "--time-window",
          "3600",
          "--max-cycle-length",
          "5",
      },
      status);

  if (output.find("CUDA support is not compiled") == std::string::npos) {
    GTEST_SKIP() << "CUDA support is compiled in this test build";
  }

  EXPECT_NE(status, 0);
  EXPECT_NE(output.find("CUDA support is not compiled"), std::string::npos);
}

TEST(CliCudaTest, TemporalReportsUnavailableBackendWhenNotCompiled) {
  int status = -1;
  const std::string output = run_cli(
      {
          "--input",
          data_path("sample_temporal.txt"),
          "--backend",
          "cuda",
          "--cuda-device",
          "0",
          "--algorithm",
          "johnson",
          "--mode",
          "temporal",
          "--time-window",
          "3600",
          "--max-cycle-length",
          "5",
      },
      status);

  if (output.find("CUDA support is not compiled") == std::string::npos) {
    GTEST_SKIP() << "CUDA support is compiled in this test build";
  }

  EXPECT_NE(status, 0);
  EXPECT_NE(output.find("CUDA support is not compiled"), std::string::npos);
}

TEST(CliCudaTest, RejectsUnknownScheduler) {
  int status = -1;
  const std::string output = run_cli(
      {
          "--input",
          data_path("sample_temporal.txt"),
          "--backend",
          "cuda",
          "--algorithm",
          "johnson",
          "--mode",
          "simple",
          "--max-cycle-length",
          "4",
          "--cuda-scheduler",
          "bogus",
      },
      status);

  EXPECT_NE(status, 0);
  EXPECT_NE(output.find("unknown cuda scheduler"), std::string::npos);
}

TEST(CliCudaTest, AcceptsWorkQueueScheduler) {
  int status = -1;
  const std::string output = run_cli(
      {
          "--input",
          data_path("sample_temporal.txt"),
          "--backend",
          "cuda",
          "--algorithm",
          "johnson",
          "--mode",
          "simple",
          "--max-cycle-length",
          "4",
          "--cuda-scheduler",
          "work-queue",
      },
      status);

  // The flag parses; without a compiled CUDA backend the run still fails at
  // dispatch with the unavailable-backend message rather than an option error.
  if (output.find("CUDA support is not compiled") == std::string::npos) {
    GTEST_SKIP() << "CUDA support is compiled in this test build";
  }
  EXPECT_NE(status, 0);
  EXPECT_EQ(output.find("unknown cuda scheduler"), std::string::npos);
  EXPECT_NE(output.find("CUDA support is not compiled"), std::string::npos);
}

}  // namespace
