cmake_minimum_required(VERSION 3.10)

project(VisionSystemTests)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(common_tests
  ../common/tests/logger_test.cpp
)

target_link_libraries(common_tests PRIVATE
  logger
  gtest_main
  gmock_main
)

include(GoogleTest)
gtest_discover_tests(common_tests)
