list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
include(CUDAConfig)
include(RunFLATC)

set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING "" FORCE)

add_executable(vision-system
   src/main.cc
   src/system_container.cc
   src/zed.cc
   src/zed_publisher.cc
)

enable_cuda_for_target(vision-system)
compile_fbs_for_target(vision-system)

find_package(OpenCV 4.10.0 REQUIRED)
message(STATUS "OpenCV library status:")
message(STATUS "    config: ${OpenCV_DIR}")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

if(CUDA_AVAILABLE)
    if(MSVC)
        target_compile_options(vision-system PRIVATE /wd4201 /wd4100 /wd4251 /wd4305 /wd5054)
    endif()
    message("-- Using ZED Camera")
    find_package(ZED 4 REQUIRED)
    include_directories(${ZED_INCLUDE_DIRS})
    link_directories(${ZED_LIBRARY_DIR})
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
            ${ZED_LIBS}
            ${OpenCV_LIBS}
    )
else()
    target_link_libraries(vision-system
        PRIVATE
            common_lib
            ${OpenCV_LIBS}
    )
endif()

set_target_warnings(vision-system)
