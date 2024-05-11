cmake_minimum_required(VERSION 3.10)

project(vision-system VERSION 1.0 DESCRIPTION "Vision System Project" LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_subdirectory(common)

# add_subdirectory(vision-system)
# add_subdirectory(vision-client)
# add_subdirectory(vision-communication)

include(FetchContent)
set(FETCHCONTENT_QUIET FALSE)
set(FETCHCONTENT_UPDATES_DISCONNECTED TRUE)

FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
  DOWNLOAD_EXTRACT_TIMESTAMP true
)

set(FETCHCONTENT_QUIET TRUE)
FetchContent_MakeAvailable(googletest)

enable_testing()
add_subdirectory(common/tests)
