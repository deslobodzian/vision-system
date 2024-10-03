include(FetchContent)

FetchContent_Declare(
  libzmq
  GIT_REPOSITORY https://github.com/zeromq/libzmq.git
  GIT_TAG v4.3.5
)

set(ZMQ_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(WITH_PERF_TOOL OFF CACHE BOOL "" FORCE)
set(ENABLE_CPACK OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED OFF CACHE BOOL "" FORCE)
set(BUILD_STATIC ON CACHE BOOL "" FORCE)

# Override libzmq cmake
function(cmake_minimum_required)
endfunction()

FetchContent_MakeAvailable(libzmq)
# Go back to default
macro(cmake_minimum_required)
  _cmake_minimum_required(${ARGN})
endmacro()

if(TARGET libzmq-static)
    if(MSVC)
        target_compile_options(libzmq-static PRIVATE /W0)
    else()
        target_compile_options(libzmq-static PRIVATE -w)
    endif()
    set_target_properties(libzmq-static PROPERTIES COMPILE_DEFINITIONS "ZMQ_STATIC")
    # alias libzmq for compatibility
    add_library(libzmq ALIAS libzmq-static)
endif()

FetchContent_Declare(
  flatbuffers
  GIT_REPOSITORY https://github.com/google/flatbuffers.git
  GIT_TAG v24.3.25
)
set(FLATBUFFERS_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(FLATBUFFERS_INSTALL OFF CACHE BOOL "" FORCE)
set(FLATBUFFERS_STRICT_MODE ON CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(flatbuffers)

add_library(common_lib STATIC src/task.cc)
target_include_directories(common_lib PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(common_lib PUBLIC libzmq)
target_compile_definitions(common_lib PRIVATE ZMQ_STATIC)

target_include_directories(common_lib PUBLIC ${flatbuffers_SOURCE_DIR}/include)

add_library(logger INTERFACE)
target_include_directories(logger INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)

set_target_warnings(common_lib)