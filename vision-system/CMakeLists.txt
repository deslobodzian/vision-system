list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
include(CUDAConfig)

add_executable(vision-system
   src/main.cc
   src/system_container.cc
   src/zed.cc
)

enable_cuda_for_target(vision-system)

if(CUDA_AVAILABLE)
    # ZED Package warnings
    if(MSVC)
        target_compile_options(vision-system PRIVATE /wd4201 /wd4100 /wd4251 /wd4305)
    endif()
    message("-- Using ZED Camera")
    find_package(ZED 4 REQUIRED)
    include_directories(${ZED_INCLUDE_DIRS})
    link_directories(${ZED_LIBRARY_DIR})
    set(ZED_LIBS ${ZED_LIBRARIES})
endif()

target_include_directories(vision-system
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src 
        ${CMAKE_SOURCE_DIR}/common/include 
)

if(CUDA_AVAILABLE)
    target_link_libraries(vision-system
        PRIVATE
            common_lib
            ${ZED_LIBS}
    )
else()
    target_link_libraries(vision-system
        PRIVATE
            common_lib
    )
endif()

set_target_warnings(vision-system)