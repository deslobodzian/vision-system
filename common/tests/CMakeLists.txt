project(common_tests)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
include(CUDAConfig)

find_package(Catch2 CONFIG REQUIRED)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

list(APPEND CMAKE_MODULE_PATH "${CATCH2_EXTRAS_DIR}")
include(Catch)

add_executable(common_tests
  logger_test.cc
  task_test.cc
  zmq_tests.cc
  device_tests.cc
  image_tests.cc
)

target_link_libraries(common_tests
  PRIVATE
  logger
  common_lib
  Catch2::Catch2WithMain
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

enable_cuda_for_target(common_tests)
catch_discover_tests(common_tests)