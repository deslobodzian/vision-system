list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
include(CUDAConfig)
include(FetchContent)

set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING "" FORCE)

add_executable(vision-system
   src/main.cc
   src/system_container.cc
   src/zed.cc
   src/cv_camera.cc
   src/tag_detector.cc
)

enable_cuda_for_target(vision-system)

find_package(OpenCV 4.12.0 REQUIRED)
message(STATUS "OpenCV library status:")
message(STATUS "    config: ${OpenCV_DIR}")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

find_package(wpilib REQUIRED)

# Need to disable shared libs, apriltags is breaking catch2 test discovery, dont know why?
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(BUILD_PYTHON_WRAPPER OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
    apriltag
    GIT_REPOSITORY https://github.com/AprilRobotics/apriltag.git
    GIT_TAG v3.4.5
)

FetchContent_MakeAvailable(apriltag)

# set(BUILD_SHARED_LIBS ON CACHE BOOL "" FORCE)

if(CUDA_AVAILABLE)
    if(MSVC)
        target_compile_options(vision-system PRIVATE 
            /wd4201 /wd4100 /wd4251 /wd4305 /wd5054
            /wd4244  # conversion warnings from apriltag
            /wd4996  # deprecated function warnings from apriltag
        )
    endif()
    message("-- Using ZED Camera")
    find_package(ZED REQUIRED)
    target_include_directories(vision-system PRIVATE ${ZED_INCLUDE_DIRS})
    target_link_directories(vision-system PRIVATE ${ZED_LIBRARY_DIR})
    
    set(ZED_LIBS ${ZED_LIBRARIES})
endif()

target_include_directories(vision-system
    PRIVATE
        ${OpenCV_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/common/include
)

if(CUDA_AVAILABLE)
    target_link_libraries(vision-system
        PRIVATE
            common_lib
            ntcore
            cscore
            ${ZED_LIBS}
            ${OpenCV_LIBS}
            apriltag
    )
else()
    target_link_libraries(vision-system
        PRIVATE
            common_lib
            ntcore
            cscore
            ${OpenCV_LIBS}
            apriltag
    )
endif()

set_target_warnings(vision-system)