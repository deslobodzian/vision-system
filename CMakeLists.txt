# Disable in-source builds to prevent source tree corruption
if("${CMAKE_CURRENT_SOURCE_DIR}" STREQUAL "${CMAKE_CURRENT_BINARY_DIR}")
    message(
        FATAL_ERROR
        "
FATAL: In-source builds are not allowed.
       You should create a separate directory for build files.
"
    )
endif()

cmake_minimum_required(VERSION 3.21)

set(CMAKE_MODULE_PATH
    ${CMAKE_MODULE_PATH}
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/"
)

project(vision-system VERSION 1.0 DESCRIPTION "Vision System Project" LANGUAGES CXX)

function(set_target_warnings target_name)
    if(MSVC)
        target_compile_options(${target_name} PRIVATE /EHsc /W4 /WX)
    else()
        target_compile_options(${target_name} PRIVATE -Wall -Wextra -pedantic -Werror)
    endif()
endfunction()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set the defualt to release with debug info (relase mode with optimization but still have debug info)
if(NOT DEFINED CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "")
    set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING "Choose the type of build." FORCE)
endif()

if(MSVC)
    if(NOT CMAKE_BUILD_TYPE)
        set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING "" FORCE)
    endif()
    add_compile_options(/MD)  # Force the runtime library to match
endif()

# Set the allowed values for CMAKE_BUILD_TYPE (optional)
set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "RelWithDebInfo" "Release" "MinSizeRel" "Debug")

if (NOT CMAKE_EXPORT_COMPILE_COMMANDS)
  set(CMAKE_EXPORT_COMPILE_COMMANDS "YES" CACHE BOOL "" FORCE)
endif()

if (WIN32)
  set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
endif()

add_subdirectory(third_party)
add_subdirectory(common)
add_subdirectory(vision-system)

enable_testing()
add_subdirectory(common/tests)
