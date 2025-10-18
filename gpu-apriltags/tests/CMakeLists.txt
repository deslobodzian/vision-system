project(gpu_apriltag_tests)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
include(CUDAConfig)

find_package(Catch2 CONFIG REQUIRED)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

list(APPEND CMAKE_MODULE_PATH "${CATCH2_EXTRAS_DIR}")
include(Catch)

add_executable(gpu_apriltag_tests
    test_test.cc
)

target_link_libraries(gpu_apriltag_tests
  PRIVATE
  logger
  common_lib
  gpu_apriltags
  Catch2::Catch2WithMain
)

target_include_directories(gpu_apriltag_tests
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

enable_cuda_for_target(gpu_apriltag_tests)
catch_discover_tests(gpu_apriltag_tests)