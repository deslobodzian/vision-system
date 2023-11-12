cmake_minimum_required(VERSION 3.5)
project(Vision_System)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

add_compile_options(-pthread -O3)

set(LOG_LEVEL "INFO" CACHE STRING "Set the logger level")

if(LOG_LEVEL STREQUAL "DEBUG")
    add_definitions(-DLOG_LEVEL=2)
elseif(LOG_LEVEL STREQUAL "INFO")
    add_definitions(-DLOG_LEVEL=1)
elseif(LOG_LEVEL STREQUAL "ERROR")
    add_definitions(-DLOG_LEVEL=0)
else()
    message(FATAL_ERROR "Unknown log level: ${LOG_LEVEL}")
endif()
message(STATUS "Using Log level ${LOG_LEVEL}")

find_package(OpenCV REQUIRED)
message(STATUS "OpenCV include dir: ${OpenCV_INCLUDE_DIRS}")
# Defualt to ONNXRuntime CPU and use CUDA later if available
find_package(onnxruntime CONFIG REQUIRED)
# message(STATUS "ONNXRuntime include dir: ${onnxruntime_INCLUDE_DIRS}")

if(APPLE)
    message(STATUS "Configuring for MacBook")
    add_definitions(-DAPPLE)
elseif(UNIX AND NOT APPLE)
    message(STATUS "Configuring for Linux")
    
    # Check for CUDA using the language support
    enable_language(CUDA)
    find_package(CUDA)
    if(CUDA_FOUND)
        add_definitions(-DLINUX_WITH_CUDA)
        message(STATUS "CUDA found: Compiling for Linux with GPU support")
        
        find_package(ZED 4 REQUIRED)
        include_directories(${ZED_INCLUDE_DIRS})
        include_directories(${CUDA_INCLUDE_DIRS})
        link_directories(${ZED_LIBRARY_DIR})
        link_directories(${CUDA_LIBRARY_DIRS})
    else()
        add_definitions(-DLINUX_WITHOUT_CUDA)
        message(STATUS "No CUDA found: Compiling for Linux without GPU support")
    endif()
endif()

include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${onnxruntime_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/include
)

link_directories(
    ${OpenCV_LIBRARY_DIRS}
)

file(GLOB_RECURSE SRC_FILES src/*.cpp)
file(GLOB_RECURSE HDR_FILES include/*.h include/*.hpp)

add_executable(${PROJECT_NAME} ${SRC_FILES})

target_link_libraries(${PROJECT_NAME} PRIVATE
    ${OpenCV_LIBRARIES}
    onnxruntime::onnxruntime
)

if(UNIX AND NOT APPLE AND CUDA_FOUND)
    target_link_libraries(${PROJECT_NAME} PRIVATE
        ${CUDA_LIBRARIES} ${CUDA_CUDART_LIBRARY}
        ${ZED_LIBRARIES}
    )
endif()