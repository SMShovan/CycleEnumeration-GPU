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
include(cmake/Doxygen.cmake)

add_library(cycle_enum_core
  src/core/graph_builder.cpp
  src/core/graph.cpp
  src/core/histogram.cpp
  src/core/options.cpp
  src/core/result.cpp
  src/core/timestamp.cpp
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

add_library(cycle_enum_sequential
  src/sequential/bruteforce.cpp
  src/sequential/cycle_union.cpp
  src/sequential/johnson.cpp
  src/sequential/read_tarjan.cpp
  src/sequential/temporal_johnson.cpp
  src/sequential/temporal_read_tarjan.cpp
)

add_library(cycle_enum::sequential ALIAS cycle_enum_sequential)

target_link_libraries(cycle_enum_sequential
  PUBLIC
    cycle_enum::core
)

target_compile_features(cycle_enum_sequential PUBLIC cxx_std_17)

target_include_directories(cycle_enum_sequential
  PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  target_compile_options(cycle_enum_sequential PRIVATE -Wall -Wextra -Wpedantic)
elseif(MSVC)
  target_compile_options(cycle_enum_sequential PRIVATE /W4)
endif()

cycle_enum_configure_sanitizers(cycle_enum_sequential)

add_library(cycle_enum_openmp
  src/openmp/openmp_config.cpp
  src/openmp/openmp_johnson.cpp
  src/openmp/openmp_read_tarjan.cpp
)

add_library(cycle_enum::openmp ALIAS cycle_enum_openmp)

target_link_libraries(cycle_enum_openmp
  PUBLIC
    cycle_enum::core
)

target_compile_features(cycle_enum_openmp PUBLIC cxx_std_17)

target_include_directories(cycle_enum_openmp
  PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  target_compile_options(cycle_enum_openmp PRIVATE -Wall -Wextra -Wpedantic)
elseif(MSVC)
  target_compile_options(cycle_enum_openmp PRIVATE /W4)
endif()

add_executable(cycle_enum_cli
  src/cli/cycle_enum_main.cpp
)

set_target_properties(cycle_enum_cli
  PROPERTIES
    OUTPUT_NAME cycle-enum
)

target_link_libraries(cycle_enum_cli
  PRIVATE
    cycle_enum::sequential
)

target_compile_features(cycle_enum_cli PRIVATE cxx_std_17)

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  target_compile_options(cycle_enum_cli PRIVATE -Wall -Wextra -Wpedantic)
elseif(MSVC)
  target_compile_options(cycle_enum_cli PRIVATE /W4)
endif()

cycle_enum_configure_sanitizers(cycle_enum_cli)

set(CYCLE_ENUM_OPENMP_AVAILABLE OFF)
if(CYCLE_ENUM_ENABLE_OPENMP)
  find_package(OpenMP QUIET)
  if(OpenMP_CXX_FOUND)
    set(CYCLE_ENUM_OPENMP_AVAILABLE ON)
    target_link_libraries(cycle_enum_openmp PUBLIC OpenMP::OpenMP_CXX)
    message(STATUS "OpenMP support enabled")
  else()
    message(WARNING "OpenMP was requested but no OpenMP CXX runtime was found; OpenMP backends will report unavailable.")
  endif()
endif()

target_compile_definitions(cycle_enum_openmp
  PUBLIC
    CYCLE_ENUM_OPENMP_ENABLED=$<BOOL:${CYCLE_ENUM_OPENMP_AVAILABLE}>
)

cycle_enum_configure_sanitizers(cycle_enum_openmp)

if(CYCLE_ENUM_ENABLE_CUDA)
  message(STATUS "CUDA option is enabled; CUDA targets will be added with the CUDA infrastructure phase.")
endif()

if(CYCLE_ENUM_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

cycle_enum_configure_doxygen()
