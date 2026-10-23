cmake_minimum_required(VERSION 3.20)
project(mlhpc LANGUAGES CXX)

set(MLHPC_CASE "A" CACHE STRING "Coupling case: A or B")
set_property(CACHE MLHPC_CASE PROPERTY STRINGS A B)

if(NOT MLHPC_CASE MATCHES "^(A|B)$")
  message(FATAL_ERROR "MLHPC_CASE must be A or B")
endif()

option(MLHPC_ENABLE_OPENMP "Enable OpenMP" ON)
option(MLHPC_FEATURE_DOUBLE "Use double-precision particle features" OFF)
option(MLHPC_ID_32BIT "Use 32-bit particle IDs" OFF)

set(MLHPC_BACKEND "DUMMY" CACHE STRING "Inference backend: DUMMY or ONNX")
set_property(CACHE MLHPC_BACKEND PROPERTY STRINGS DUMMY ONNX)

if(NOT MLHPC_BACKEND MATCHES "^(DUMMY|ONNX)$")
  message(FATAL_ERROR "MLHPC_BACKEND must be DUMMY or ONNX")
endif()

option(MLHPC_DOWNLOAD_ONNX "Download a private CPU ONNX Runtime" OFF)

option(ENABLE_TESTING "Build and register tests" ON)

set(MLHPC_ONNX_VERSION "1.31.0" CACHE STRING "ONNX Runtime download version")
set(MLHPC_ONNX_ROOT "" CACHE PATH "Existing ONNX Runtime installation")
set(MLHPC_ONNX_MODEL "" CACHE FILEPATH "rho/u compatibility model for ONNX tests")
set(MLHPC_MPI_TEST_RANKS 4 CACHE STRING "Ranks for each MPI test")

if(MLHPC_DOWNLOAD_ONNX AND NOT MLHPC_BACKEND STREQUAL "ONNX")
  message(FATAL_ERROR "Downloading ONNX Runtime requires MLHPC_BACKEND=ONNX")
endif()


find_package(MPI REQUIRED COMPONENTS CXX)

set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

add_library(
    mlhpc_options # Shared C++ standard and warning settings
    INTERFACE
)
target_compile_features(
    mlhpc_options 
    INTERFACE 
    cxx_std_20
)

if(CMAKE_CXX_COMPILER_ID MATCHES "^(GNU|Clang)$")
    target_compile_options(
        mlhpc_options 
        INTERFACE
        -Wall -Wextra -Wpedantic -Wcast-align -Wshadow -Wold-style-cast
        -Wnon-virtual-dtor -Woverloaded-virtual
        -Wnull-dereference -Wdouble-promotion
    )
endif()

add_library(
    mlhpc_core # Snapshot I/O and coupling code
    STATIC
    src/io_gadget2.cpp
    src/partition.cpp
    src/group_distribution.cpp
    src/format2_distributor.cpp
    src/units.cpp
    src/coupling_async_mpi.cpp
    src/coupling_service.cpp
    src/coupling_${MLHPC_CASE}.cpp
)

target_include_directories(
    mlhpc_core 
    PUBLIC
    "${PROJECT_SOURCE_DIR}/include"
)

target_link_libraries(
    mlhpc_core 
    PUBLIC
    mlhpc_options
    MPI::MPI_CXX
)

target_compile_definitions(
    mlhpc_core 
    PUBLIC
    MLHPC_CASE_${MLHPC_CASE}=1
    OMPI_SKIP_MPICXX=1
    MPICH_SKIP_MPICXX=1
)

if(MLHPC_FEATURE_DOUBLE)
  target_compile_definitions(mlhpc_core PUBLIC MLHPC_FEATURE_DOUBLE=1)
endif()

if(MLHPC_ID_32BIT)
  target_compile_definitions(mlhpc_core PUBLIC MLHPC_ID_32BIT=1)
endif()

if(MLHPC_ENABLE_OPENMP)
  find_package(OpenMP REQUIRED COMPONENTS CXX)
  target_link_libraries(mlhpc_core PUBLIC OpenMP::OpenMP_CXX)
endif()

if(MLHPC_BACKEND STREQUAL "DUMMY" OR ENABLE_TESTING)
  add_library(mlhpc_dummy STATIC src/inference_dummy.cpp)
  target_link_libraries(mlhpc_dummy PUBLIC mlhpc_core)
endif()

add_library(mlhpc_backend STATIC src/inference_factory.cpp)
target_link_libraries(mlhpc_backend PUBLIC mlhpc_core)

if(MLHPC_BACKEND STREQUAL "ONNX")
    include(cmake/OnnxRuntime.cmake)

    add_library(
        mlhpc_onnx # Optional ONNX engine implementation
        STATIC 
        src/inference_onnx.cpp
    )

    target_link_libraries(
        mlhpc_onnx # Only src/inference_onnx.cpp depends on ONNXRuntime
        PUBLIC mlhpc_core
        PRIVATE ONNXRuntime::ONNXRuntime
    )

  target_compile_definitions(mlhpc_backend PRIVATE MLHPC_BACKEND_ONNX=1)
  target_link_libraries(mlhpc_backend PRIVATE mlhpc_onnx)
else()
  target_compile_definitions(mlhpc_backend PRIVATE MLHPC_BACKEND_ONNX=0)
  target_link_libraries(mlhpc_backend PRIVATE mlhpc_dummy)
endif()

add_executable(
    mlhpc # Main executable for the project
    src/main.cpp
)
target_link_libraries(mlhpc PRIVATE mlhpc_backend)

if(ENABLE_TESTING)
  enable_testing()
  add_subdirectory(tests)
endif()