set(backend_test_arguments)
set(backend_smoke_arguments)
if(MLHPC_BACKEND STREQUAL "ONNX")
  if(NOT MLHPC_ONNX_MODEL OR NOT EXISTS "${MLHPC_ONNX_MODEL}")
    message(FATAL_ERROR
      "Set MLHPC_ONNX_MODEL to the rho/u compatibility model.onnx")
  endif()
  list(APPEND backend_test_arguments "${MLHPC_ONNX_MODEL}")
  list(APPEND backend_smoke_arguments "--model=${MLHPC_ONNX_MODEL}")
endif()

foreach(name IN ITEMS
    test_header_layout
    test_io_gadget2
    test_partition
    test_group_distribution
    test_units
    test_inference_dummy)

  add_executable(${name} ${name}.cpp)
  if(name STREQUAL "test_inference_dummy")
    target_link_libraries(${name} PRIVATE mlhpc_dummy)
  else()
    target_link_libraries(${name} PRIVATE mlhpc_core)
  endif()

  add_test(NAME ${name} COMMAND ${name})
  set_tests_properties(${name} PROPERTIES 
        LABELS unit # no MPI launcher required
    )
endforeach()

foreach(name IN ITEMS
    test_format2_distributor
    test_coupling_A
    test_coupling_B)

  add_executable(${name} ${name}.cpp)
  set(test_arguments)
  if(name MATCHES "^test_coupling_[AB]$")
    target_link_libraries(${name} PRIVATE mlhpc_backend mlhpc_dummy)
    set(test_arguments ${backend_test_arguments})
  else()
    target_link_libraries(${name} PRIVATE mlhpc_core)
  endif()

  add_test(NAME ${name}
    COMMAND "${MPIEXEC_EXECUTABLE}" "${MPIEXEC_NUMPROC_FLAG}"
      "${MLHPC_MPI_TEST_RANKS}" ${MPIEXEC_PREFLAGS}
      $<TARGET_FILE:${name}> ${MPIEXEC_POSTFLAGS} ${test_arguments})

  set_tests_properties(${name} PROPERTIES
    LABELS mpi
    PROCESSORS "${MLHPC_MPI_TEST_RANKS}"
    TIMEOUT 60)
endforeach()

add_test(NAME test_backend_smoke
  COMMAND "${MPIEXEC_EXECUTABLE}" "${MPIEXEC_NUMPROC_FLAG}"
    "${MLHPC_MPI_TEST_RANKS}" ${MPIEXEC_PREFLAGS}
    $<TARGET_FILE:mlhpc> ${MPIEXEC_POSTFLAGS} ${backend_smoke_arguments})
set_tests_properties(test_backend_smoke PROPERTIES
  LABELS mpi
  PROCESSORS "${MLHPC_MPI_TEST_RANKS}"
  TIMEOUT 60)

if(MLHPC_BACKEND STREQUAL "ONNX")
  add_executable(test_inference_onnx test_inference_onnx.cpp)
  target_link_libraries(test_inference_onnx PRIVATE mlhpc_onnx mlhpc_dummy)

  add_test(NAME test_inference_onnx
    COMMAND test_inference_onnx "${MLHPC_ONNX_MODEL}")

  set_tests_properties(test_inference_onnx PROPERTIES
    LABELS onnx
    TIMEOUT 30)
endif()
