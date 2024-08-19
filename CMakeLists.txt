cmake_minimum_required(VERSION 3.14...3.21)

project(vision-system VERSION 1.0 DESCRIPTION "Vision System Project" LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(MSVC)
  add_compile_options(/W4 /WX)
else()
  add_compile_options(-Wall -Wextra -pedantic -Werror)
endif()

# Set the defualt to release with debug info (relase mode with optimization but still have debug info)
if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING "" FORCE)
endif()

if (NOT CMAKE_EXPORT_COMPILE_COMMANDS) 
  set(CMAKE_EXPORT_COMPILE_COMMANDS "YES" CACHE BOOL "" FORCE)
endif()

if (WIN32)
  set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
endif()

include(FetchContent)
set(FETCHCONTENT_QUIET FALSE)
set(FETCHCONTENT_UPDATES_DISCONNECTED TRUE)

# Fetch Google Test
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
  DOWNLOAD_EXTRACT_TIMESTAMP true
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

add_subdirectory(common)
# add_subdirectory(vision-system)
# add_subdirectory(vision-client)
# add_subdirectory(vision-communication)

# Enable testing
enable_testing()
add_subdirectory(common/tests)
