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

add_executable(cycle_enum_smoke_test
  unit/smoke_test.cpp
)

target_link_libraries(cycle_enum_smoke_test
  PRIVATE
    cycle_enum::core
    GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(cycle_enum_smoke_test)

