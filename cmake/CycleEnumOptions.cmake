# Project-wide build switches.
#
# This file intentionally keeps options centralized so future implementation
# phases can add OpenMP, CUDA, tests, benchmarks, and documentation without
# changing the top-level CMake interface.

option(CYCLE_ENUM_BUILD_TESTS "Build unit and integration tests" ON)
option(CYCLE_ENUM_BUILD_BENCHMARKS "Build benchmark executables" OFF)
option(CYCLE_ENUM_ENABLE_OPENMP "Enable OpenMP CPU implementations" OFF)
option(CYCLE_ENUM_ENABLE_CUDA "Enable CUDA GPU implementations" OFF)
option(CYCLE_ENUM_ENABLE_DOXYGEN "Enable Doxygen documentation targets" OFF)
option(CYCLE_ENUM_ENABLE_SANITIZERS "Enable address and undefined behavior sanitizers for supported compilers" OFF)

