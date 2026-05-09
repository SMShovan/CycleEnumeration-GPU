cmake_minimum_required(VERSION 3.20)

project(
  CycleEnumerationGPU
  VERSION 0.1.0
  DESCRIPTION "Single-node CPU and CUDA cycle enumeration"
  LANGUAGES CXX
)

include(GNUInstallDirs)
include(cmake/CycleEnumOptions.cmake)
include(cmake/Sanitizers.cmake)

add_library(cycle_enum_core
  src/core/version.cpp
)

add_library(cycle_enum::core ALIAS cycle_enum_core)

target_compile_features(cycle_enum_core PUBLIC cxx_std_17)

target_include_directories(cycle_enum_core
  PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_compile_definitions(cycle_enum_core
  PUBLIC
    CYCLE_ENUM_VERSION_MAJOR=${PROJECT_VERSION_MAJOR}
    CYCLE_ENUM_VERSION_MINOR=${PROJECT_VERSION_MINOR}
    CYCLE_ENUM_VERSION_PATCH=${PROJECT_VERSION_PATCH}
)

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  target_compile_options(cycle_enum_core PRIVATE -Wall -Wextra -Wpedantic)
elseif(MSVC)
  target_compile_options(cycle_enum_core PRIVATE /W4)
endif()

cycle_enum_configure_sanitizers(cycle_enum_core)

if(CYCLE_ENUM_ENABLE_OPENMP)
  message(STATUS "OpenMP option is enabled; OpenMP targets will be added with the OpenMP implementation phase.")
endif()

if(CYCLE_ENUM_ENABLE_CUDA)
  message(STATUS "CUDA option is enabled; CUDA targets will be added with the CUDA infrastructure phase.")
endif()

if(CYCLE_ENUM_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

if(CYCLE_ENUM_ENABLE_DOXYGEN)
  message(STATUS "Doxygen option is enabled; documentation targets will be added in the Doxygen setup phase.")
endif()
