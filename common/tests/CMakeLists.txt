cmake_minimum_required(VERSION 3.10)

project(common_tests)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(common_tests
  logger_test.cpp
  task_test.cpp
  zmq_publisher_test.cpp
)

target_link_libraries(common_tests PRIVATE
  logger
  common_lib
  gtest_main
  gmock_main
)

target_include_directories(common_tests PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

include(GoogleTest)
gtest_discover_tests(common_tests)
