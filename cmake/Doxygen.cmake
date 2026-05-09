# Doxygen integration.
#
# Documentation is optional because build machines and CI runners may not have
# Doxygen installed. When enabled and available, the target `cycle_enum_docs`
# generates HTML API documentation under `docs/doxygen/html`.

function(cycle_enum_configure_doxygen)
  if(NOT CYCLE_ENUM_ENABLE_DOXYGEN)
    return()
  endif()

  find_package(Doxygen QUIET)
  if(NOT DOXYGEN_FOUND)
    message(WARNING "CYCLE_ENUM_ENABLE_DOXYGEN is ON, but Doxygen was not found.")
    return()
  endif()

  set(doxygen_input_file "${PROJECT_SOURCE_DIR}/docs/doxygen/Doxyfile.in")
  set(doxygen_output_file "${PROJECT_BINARY_DIR}/Doxyfile")

  configure_file(
    "${doxygen_input_file}"
    "${doxygen_output_file}"
    @ONLY
  )

  add_custom_target(cycle_enum_docs
    COMMAND "${DOXYGEN_EXECUTABLE}" "${doxygen_output_file}"
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
    COMMENT "Generating CycleEnumeration-GPU API documentation"
    VERBATIM
  )
endfunction()

