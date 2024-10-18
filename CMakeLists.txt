cmake_minimum_required(VERSION 3.14...3.21)

project(vision-system VERSION 1.0 DESCRIPTION "Vision System Project" LANGUAGES CXX)

function(set_target_warnings target_name)
    if(MSVC)
        target_compile_options(${target_name} PRIVATE /W4 /WX)
    else()
        target_compile_options(${target_name} PRIVATE -Wall -Wextra -pedantic -Werror)
    endif()
endfunction()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set the defualt to release with debug info (relase mode with optimization but still have debug info)
if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING "" FORCE)
endif()

if (NOT CMAKE_EXPORT_COMPILE_COMMANDS)
  set(CMAKE_EXPORT_COMPILE_COMMANDS "YES" CACHE BOOL "" FORCE)
endif()

function(enable_cuda_for_target target)
    if(CUDA_AVAILABLE)
        set_target_properties(${target} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
        target_compile_definitions(${target} PRIVATE CUDA)
        set_target_properties(${target} PROPERTIES CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES})
    else()
        message(WARNING "CUDA not available, skipping CUDA setup for target ${target}")
    endif()
endfunction()

if (WIN32)
  set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
endif()

include(FetchContent)
set(FETCHCONTENT_QUIET FALSE)
set(FETCHCONTENT_UPDATES_DISCONNECTED TRUE)

FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.15.2
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
