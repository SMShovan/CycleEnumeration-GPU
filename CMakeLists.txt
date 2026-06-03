cmake_minimum_required(VERSION 3.20)

project(
  CycleEnumerationGPU
  VERSION 0.1.0
  DESCRIPTION "Single-node CPU and CUDA cycle enumeration"
  LANGUAGES CXX
)

include(GNUInstallDirs)
include(CheckLanguage)
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
  src/openmp/openmp_read_tarjan_tasks.cpp
  src/openmp/openmp_temporal_johnson.cpp
  src/openmp/openmp_temporal_read_tarjan.cpp
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

add_library(cycle_enum_dynamic
  src/dynamic/batch_generator.cpp
  src/dynamic/edge_change.cpp
  src/engine/histogram_engine.cpp
)

add_library(cycle_enum::dynamic ALIAS cycle_enum_dynamic)

target_link_libraries(cycle_enum_dynamic
  PUBLIC
    cycle_enum::core
    cycle_enum::sequential
)

target_compile_features(cycle_enum_dynamic PUBLIC cxx_std_17)

target_include_directories(cycle_enum_dynamic
  PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  target_compile_options(cycle_enum_dynamic PRIVATE -Wall -Wextra -Wpedantic)
elseif(MSVC)
  target_compile_options(cycle_enum_dynamic PRIVATE /W4)
endif()

cycle_enum_configure_sanitizers(cycle_enum_dynamic)

add_executable(cycle_enum_cli
  src/cli/cycle_enum_main.cpp
)

set_target_properties(cycle_enum_cli
  PROPERTIES
    OUTPUT_NAME cycle-enum
)

target_link_libraries(cycle_enum_cli
  PRIVATE
    cycle_enum::cuda
    cycle_enum::dynamic
    cycle_enum::openmp
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

set(CYCLE_ENUM_CUDA_AVAILABLE OFF)
if(CYCLE_ENUM_ENABLE_CUDA)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    set(CYCLE_ENUM_CUDA_AVAILABLE ON)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
      set(CMAKE_CUDA_ARCHITECTURES 90)
    endif()
    message(STATUS "CUDA support enabled for architectures: ${CMAKE_CUDA_ARCHITECTURES}")
  else()
    message(WARNING "CUDA was requested but no CUDA compiler was found; CUDA backends will report unavailable.")
  endif()
endif()

add_library(cycle_enum_cuda
  src/cuda/cuda_branch_split.cpp
  src/cuda/cuda_config.cpp
  src/cuda/cuda_graph.cpp
  src/cuda/cuda_johnson.cpp
  src/cuda/cuda_timestamp_groups.cpp
  src/cuda/cuda_work_item.cpp
  src/cuda/cuda_work_queue.cpp
)

add_library(cycle_enum::cuda ALIAS cycle_enum_cuda)

target_link_libraries(cycle_enum_cuda
  PUBLIC
    cycle_enum::core
    cycle_enum::sequential
)

target_compile_features(cycle_enum_cuda PUBLIC cxx_std_17)

target_include_directories(cycle_enum_cuda
  PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_compile_definitions(cycle_enum_cuda
  PUBLIC
    CYCLE_ENUM_CUDA_ENABLED=$<BOOL:${CYCLE_ENUM_CUDA_AVAILABLE}>
)

if(CYCLE_ENUM_CUDA_AVAILABLE)
  target_sources(cycle_enum_cuda
    PRIVATE
      src/cuda/cuda_johnson_kernel.cu
  )
  target_link_libraries(cycle_enum_cuda PUBLIC CUDA::cudart)
  set_target_properties(cycle_enum_cuda
    PROPERTIES
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED ON
  )
  # Enable NVTX scoped ranges for Nsight profiling when the NVTX target is
  # available. The ranges are no-ops without this define.
  if(TARGET CUDA::nvToolsExt)
    target_link_libraries(cycle_enum_cuda PUBLIC CUDA::nvToolsExt)
    target_compile_definitions(cycle_enum_cuda PRIVATE CYCLE_ENUM_WITH_NVTX)
    message(STATUS "NVTX profiling ranges enabled")
  endif()
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  target_compile_options(cycle_enum_cuda PRIVATE -Wall -Wextra -Wpedantic)
elseif(MSVC)
  target_compile_options(cycle_enum_cuda PRIVATE /W4)
endif()

cycle_enum_configure_sanitizers(cycle_enum_cuda)

if(CYCLE_ENUM_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

if(CYCLE_ENUM_BUILD_BENCHMARKS)
  find_package(Python3 COMPONENTS Interpreter REQUIRED)
  add_custom_target(cycle_enum_benchmark_smoke
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
      --output ${PROJECT_BINARY_DIR}/benchmarks/smoke.csv
    DEPENDS cycle_enum_cli
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    COMMENT "Running a one-case benchmark smoke test"
  )
endif()

cycle_enum_configure_doxygen()
