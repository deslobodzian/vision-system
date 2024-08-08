cmake_minimum_required(VERSION 3.14)

include(FetchContent)

# Fetch ZeroMQ
FetchContent_Declare(
  libzmq
  GIT_REPOSITORY https://github.com/zeromq/libzmq.git
  GIT_TAG v4.3.5
)
FetchContent_MakeAvailable(libzmq)

# Fetch FlatBuffers
FetchContent_Declare(
  flatbuffers
  GIT_REPOSITORY https://github.com/google/flatbuffers.git
  GIT_TAG v24.3.25
)
set(FLATBUFFERS_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(FLATBUFFERS_INSTALL OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(flatbuffers)

# Common library
add_library(common_lib STATIC src/task.cpp)
target_include_directories(common_lib PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Link against the ZeroMQ library
target_link_libraries(common_lib PUBLIC libzmq)

# Include FlatBuffers for use in the project
target_include_directories(common_lib PUBLIC ${flatbuffers_SOURCE_DIR}/include)

# Logger library
add_library(logger INTERFACE)
target_include_directories(logger INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Copy ZeroMQ DLL for targets in this directory
if(WIN32)
  add_custom_command(TARGET common_lib POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      $<TARGET_FILE:libzmq>
      $<TARGET_FILE_DIR:common_lib>
  )
endif()