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
  unit/bruteforce_test.cpp
  unit/graph_parser_test.cpp
  unit/graph_view_test.cpp
  unit/histogram_test.cpp
  unit/options_test.cpp
  unit/result_test.cpp
  unit/smoke_test.cpp
  unit/timestamp_test.cpp
)

target_link_libraries(cycle_enum_core_test
  PRIVATE
    cycle_enum::sequential
    GTest::gtest_main
)

target_compile_definitions(cycle_enum_core_test
  PRIVATE
    CYCLE_ENUM_TEST_DATA_DIR="${PROJECT_SOURCE_DIR}/tests/data"
)

include(GoogleTest)
gtest_discover_tests(cycle_enum_core_test)
