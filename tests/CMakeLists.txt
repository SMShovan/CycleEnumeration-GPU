include(FetchContent)

# Prefer a pinned GoogleTest release so test behavior is reproducible across
# the local Mac development machine and the later Linux GPU cluster.
set(INSTALL_GTEST OFF CACHE BOOL "Disable GoogleTest installation" FORCE)
set(gtest_force_shared_crt ON CACHE BOOL "Use shared CRT for GoogleTest" FORCE)

FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.14.0
)

FetchContent_MakeAvailable(googletest)

add_executable(cycle_enum_core_test
  integration/johnson_static_test.cpp
  integration/johnson_time_window_test.cpp
  integration/openmp_johnson_test.cpp
  integration/openmp_read_tarjan_test.cpp
  integration/openmp_task_parity_test.cpp
  integration/openmp_temporal_johnson_test.cpp
  integration/openmp_temporal_read_tarjan_test.cpp
  integration/path_bundling_test.cpp
  integration/cli_sequential_test.cpp
  integration/cuda_johnson_test.cpp
  integration/cuda_blocking_test.cpp
  integration/cuda_rank_test.cpp
  integration/dynamic_update_cuda_test.cpp
  integration/dynamic_update_openmp_test.cpp
  integration/dynamic_update_parity_test.cpp
  integration/read_tarjan_static_test.cpp
  integration/read_tarjan_time_window_test.cpp
  integration/sequential_parity_test.cpp
  integration/temporal_johnson_test.cpp
  integration/temporal_read_tarjan_test.cpp
  unit/batch_generator_test.cpp
  unit/bruteforce_test.cpp
  unit/cuda_branch_split_test.cpp
  unit/cuda_config_test.cpp
  unit/cuda_graph_test.cpp
  unit/cuda_profiling_test.cpp
  unit/cuda_timestamp_groups_test.cpp
  unit/cuda_timestamp_test.cpp
  unit/cuda_visited_set_test.cpp
  unit/cuda_work_item_test.cpp
  unit/cuda_work_queue_test.cpp
  unit/cycles_through_edge_test.cpp
  unit/directed_graph_test.cpp
  unit/edge_change_test.cpp
  unit/cycle_union_test.cpp
  unit/graph_parser_test.cpp
  unit/graph_view_test.cpp
  unit/histogram_engine_test.cpp
  unit/histogram_test.cpp
  unit/openmp_config_test.cpp
  unit/options_test.cpp
  unit/result_test.cpp
  unit/smoke_test.cpp
  unit/temporal_bruteforce_test.cpp
  unit/timestamp_test.cpp
)

target_link_libraries(cycle_enum_core_test
  PRIVATE
    cycle_enum::cuda
    cycle_enum::dynamic
    cycle_enum::sequential
    cycle_enum::openmp
    GTest::gtest_main
)

target_compile_definitions(cycle_enum_core_test
  PRIVATE
    CYCLE_ENUM_TEST_DATA_DIR="${PROJECT_SOURCE_DIR}/tests/data"
    CYCLE_ENUM_CLI_PATH="$<TARGET_FILE:cycle_enum_cli>"
)

add_dependencies(cycle_enum_core_test cycle_enum_cli)

include(GoogleTest)
gtest_discover_tests(cycle_enum_core_test)

find_package(Python3 COMPONENTS Interpreter QUIET)
if(Python3_Interpreter_FOUND)
  add_test(
    NAME benchmark_harness_smoke
    COMMAND
      ${Python3_EXECUTABLE}
      ${PROJECT_SOURCE_DIR}/benchmarks/run_cycle_benchmarks.py
      --cycle-enum $<TARGET_FILE:cycle_enum_cli>
      --input ${PROJECT_SOURCE_DIR}/tests/data/reference_sample.txt
      --backend sequential
      --algorithm johnson
      --mode simple-time-window
      --time-window 3600
      --max-cycle-length 6
      --warmup 0
      --repeat 1
      --output ${PROJECT_BINARY_DIR}/benchmarks/ctest-smoke.csv
  )
  set_tests_properties(benchmark_harness_smoke
    PROPERTIES
      LABELS benchmark
      WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
  )
endif()
