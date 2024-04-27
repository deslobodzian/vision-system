cmake_minimum_required(VERSION 3.10)
project(common)

set(FETCH_CONTENT_DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
cmake_policy(SET CMP0148 NEW)

add_library(logger INTERFACE)
target_include_directories(logger INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(logger_test tests/logger_test.cpp)
target_link_libraries(logger_test PRIVATE logger gtest_main gmock_main)

include(GoogleTest)
gtest_discover_tests(logger_test)
