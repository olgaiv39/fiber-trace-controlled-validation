cmake_minimum_required(VERSION 3.28)

project(fiber_trace_controlled_validation NONE)

message(STATUS "Configure the native baseline through the upstream source tree:")
message(STATUS "  cmake -S <villa-upstream-main>/volume-cartographer -B build-vc-native ")
message(STATUS "    -DCMAKE_PROJECT_TOP_LEVEL_INCLUDES=${CMAKE_CURRENT_SOURCE_DIR}/cmake/InjectNativeHarness.cmake")
