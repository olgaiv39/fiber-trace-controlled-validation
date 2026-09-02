if(NOT DEFINED FIBER_TRACE_CONTROLLED_VALIDATION_ROOT)
    get_filename_component(FIBER_TRACE_CONTROLLED_VALIDATION_ROOT "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
endif()

# VC only uses libbacktrace in its Linux crash handler. On this macOS-only
# validation build, make the optional pkg-config target available before VC's
# configuration so it does not fetch an unused third-party source tree.
if(APPLE AND NOT TARGET PkgConfig::LIBBACKTRACE)
    add_library(fiber_trace_validation_noop_libbacktrace INTERFACE)
    add_library(PkgConfig::LIBBACKTRACE ALIAS fiber_trace_validation_noop_libbacktrace)
endif()

function(fiber_trace_controlled_validation_add_native_harness)
    add_executable(native_trace_segment
        "${FIBER_TRACE_CONTROLLED_VALIDATION_ROOT}/src/native_trace_segment.cpp")
    target_link_libraries(native_trace_segment PRIVATE vc_fiber_tracer vc_lasagna)
    target_compile_features(native_trace_segment PRIVATE cxx_std_20)
endfunction()

# This file is injected while the upstream project is configured. Deferring the
# target declaration keeps the upstream source tree untouched while allowing the
# harness to link its ordinary vc_fiber_tracer/vc_lasagna targets.
cmake_language(DEFER CALL fiber_trace_controlled_validation_add_native_harness)
