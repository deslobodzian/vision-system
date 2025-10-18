include(FetchContent)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)
include(CUDAConfig)

# All part of apriltags
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE) 
set(BUILD_PYTHON_WRAPPER OFF CACHE BOOL "" FORCE)
# Examples need opencv, but I do not want this requirement in the lib
set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
    apriltag
    GIT_REPOSITORY https://github.com/AprilRobotics/apriltag.git
    GIT_TAG v3.4.5
)

FetchContent_MakeAvailable(apriltag)

add_library(gpu_apriltags STATIC src/test.cc)
target_include_directories(gpu_apriltags PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/common/include
)

target_link_libraries(gpu_apriltags PUBLIC
    common_lib
    logger
    apriltag
)

set_target_warnings(gpu_apriltags)
enable_cuda_for_target(gpu_apriltags)
