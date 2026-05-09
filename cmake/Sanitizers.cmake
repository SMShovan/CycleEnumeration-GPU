# Sanitizer helper for development builds.
#
# Sanitizers are useful while implementing the sequential and OpenMP phases.
# CUDA targets will get separate validation rules later because host sanitizer
# flags do not apply cleanly to device code.

function(cycle_enum_configure_sanitizers target_name)
  if(NOT CYCLE_ENUM_ENABLE_SANITIZERS)
    return()
  endif()

  if(MSVC)
    message(WARNING "CYCLE_ENUM_ENABLE_SANITIZERS is not configured for MSVC.")
    return()
  endif()

  if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    message(WARNING "CYCLE_ENUM_ENABLE_SANITIZERS is enabled, but the compiler is not Clang or GNU.")
    return()
  endif()

  set(sanitizer_flags "-fsanitize=address,undefined")
  target_compile_options(${target_name} PRIVATE ${sanitizer_flags} -fno-omit-frame-pointer)
  target_link_options(${target_name} PRIVATE ${sanitizer_flags})
endfunction()

