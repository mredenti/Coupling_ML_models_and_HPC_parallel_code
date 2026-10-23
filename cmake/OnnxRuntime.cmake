if(MLHPC_DOWNLOAD_ONNX)
  if(MLHPC_ONNX_ROOT)
    message(FATAL_ERROR
      "Choose MLHPC_DOWNLOAD_ONNX or MLHPC_ONNX_ROOT, not both")
  endif()

  if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux"
     OR NOT CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|AMD64)$")
    message(FATAL_ERROR
      "Automatic ONNX download supports Linux x86_64 only")
  endif()

  include(FetchContent)

  if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
  endif()

  FetchContent_Declare(onnxruntime
    URL "https://github.com/microsoft/onnxruntime/releases/download/v${MLHPC_ONNX_VERSION}/onnxruntime-linux-x64-${MLHPC_ONNX_VERSION}.tgz")

  FetchContent_MakeAvailable(onnxruntime)
  set(_ort_root "${onnxruntime_SOURCE_DIR}")
else()
  if(NOT MLHPC_ONNX_ROOT)
    message(FATAL_ERROR
      "Set MLHPC_ONNX_ROOT or enable MLHPC_DOWNLOAD_ONNX")
  endif()

  set(_ort_root "${MLHPC_ONNX_ROOT}")
endif()

set(_ort_library "${_ort_root}/lib/libonnxruntime.so")

if(NOT EXISTS "${_ort_root}/include/onnxruntime_cxx_api.h"
   OR NOT EXISTS "${_ort_library}")
  message(FATAL_ERROR "Invalid ONNX Runtime prefix: ${_ort_root}")
endif()

add_library(ONNXRuntime::ONNXRuntime SHARED IMPORTED)

set_target_properties(
    ONNXRuntime::ONNXRuntime PROPERTIES
    IMPORTED_LOCATION "${_ort_library}"
    INTERFACE_INCLUDE_DIRECTORIES "${_ort_root}/include"
)

message(STATUS "ONNX Runtime prefix: ${_ort_root}")