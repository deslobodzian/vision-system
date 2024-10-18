project(common_tests)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
include(CUDAConfig)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


add_executable(common_tests
  logger_test.cc
  task_test.cc
  zmq_tests.cc
  allocator_test.cc
  device_tests.cc
)

target_link_libraries(common_tests
  PRIVATE
  logger
  common_lib
  gtest
  gmock
  gtest_main
)

target_include_directories(common_tests
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

if(MSVC)
  add_custom_command(TARGET common_tests POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      $<TARGET_FILE:libzmq-static>
      $<TARGET_FILE_DIR:common_tests>
  )
endif()

include(GoogleTest)
gtest_discover_tests(common_tests)
enable_cuda_for_target(common_tests)
