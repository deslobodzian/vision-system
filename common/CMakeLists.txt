cmake_minimum_required(VERSION 3.10)
project(MyProject)

include(FetchContent)

# Fetch ZeroMQ
FetchContent_Declare(
  libzmq
  GIT_REPOSITORY https://github.com/zeromq/libzmq.git
  GIT_TAG v4.3.5  # Adjust the tag as needed
)
FetchContent_GetProperties(libzmq)
if(NOT libzmq_POPULATED)
  FetchContent_Populate(libzmq)
  add_subdirectory(${libzmq_SOURCE_DIR} ${libzmq_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()

# Fetch FlatBuffers
FetchContent_Declare(
  flatbuffers
  GIT_REPOSITORY https://github.com/google/flatbuffers.git
  GIT_TAG  v24.3.25  # Adjust the tag as needed
)
FetchContent_GetProperties(flatbuffers)
if(NOT flatbuffers_POPULATED)
  FetchContent_Populate(flatbuffers)
  set(FLATBUFFERS_BUILD_TESTS OFF CACHE BOOL "" FORCE)  # Optionally turn off tests and samples
  set(FLATBUFFERS_INSTALL OFF CACHE BOOL "" FORCE)      # Do not install FlatBuffers system-wide
  add_subdirectory(${flatbuffers_SOURCE_DIR} ${flatbuffers_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()

add_library(common_lib STATIC src/task.cpp)
target_include_directories(common_lib PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Link against the ZeroMQ library
target_link_libraries(common_lib PUBLIC libzmq)

# Include FlatBuffers for use in the project
target_include_directories(common_lib PUBLIC ${flatbuffers_SOURCE_DIR}/include)

add_library(logger INTERFACE)
target_include_directories(logger INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)

