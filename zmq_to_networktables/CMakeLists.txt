cmake_minimum_required(VERSION 3.10)
project(zmq_to_networktables)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++20 -pthread -Wno-psabi")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -Og")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
set(THREADS_PREFER_PTHREAD_FLAG ON)

find_package(ZeroMQ REQUIRED)
find_package(ntcore REQUIRED)
find_package(flatbuffers REQUIRED)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${ZeroMQ_INCLUDE_DIRS}
    ${flatbuffers_INCLUDE_DIR}
    ${CMAKE_BINARY_DIR}/generated_fbs
    ${CMAKE_SOURCE_DIR}/include/networking
)

file(GLOB_RECURSE ZMQ_TO_NT_SRC_FILES src/*.cpp)
file(GLOB_RECURSE ZMQ_TO_NT_HDR_FILES include/*.h include/*.hpp)

add_executable(
    ${PROJECT_NAME}
    ${ZMQ_TO_NT_SRC_FILES}
    ${ZMQ_TO_NT_HDR_FILES}
)
add_dependencies(${PROJECT_NAME} GenerateFlatBuffers)
target_link_libraries(${PROJECT_NAME} PRIVATE
    ${ZeroMQ_LIBRARY}
    ntcore
    flatbuffers::flatbuffers
)
