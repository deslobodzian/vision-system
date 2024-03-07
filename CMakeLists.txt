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

set(FLATBUFFERS_FLATC_EXECUTABLE flatc)
set(FBS_SCHEMA_DIR ${CMAKE_CURRENT_SOURCE_DIR}/fb_schemas) 
set(FBS_GENERATED_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated_fbs) 

file(GLOB_RECURSE FBS_FILES ${FBS_SCHEMA_DIR}/*.fbs) 
file(MAKE_DIRECTORY ${FBS_GENERATED_DIR})

foreach(FBS_FILE ${FBS_FILES})
    get_filename_component(FBS_NAME ${FBS_FILE} NAME_WE)
    set(GENERATED_FBS_SRC ${FBS_GENERATED_DIR}/${FBS_NAME}_generated.h)

    add_custom_command(
        OUTPUT ${GENERATED_FBS_SRC}
        COMMAND ${CMAKE_COMMAND} -E echo "Running flatc on ${FBS_FILE}"
        COMMAND ${FLATBUFFERS_FLATC_EXECUTABLE} -c -o ${FBS_GENERATED_DIR} ${FBS_FILE}
        COMMAND ${FLATBUFFERS_FLATC_EXECUTABLE} --java -o ${FBS_GENERATED_DIR} ${FBS_FILE}
        COMMAND ${FLATBUFFERS_FLATC_EXECUTABLE} --python -o ${FBS_GENERATED_DIR} ${FBS_FILE}
        DEPENDS ${FBS_FILE}
        COMMENT "Compiling ${FBS_FILE} to ${GENERATED_FBS_SRC}"
    )
    list(APPEND GENERATED_FBS_SRCS ${GENERATED_FBS_SRC})
endforeach()

add_custom_target(GenerateFlatBuffers DEPENDS ${GENERATED_FBS_SRCS})

find_package(OpenCV REQUIRED)
message(STATUS "OpenCV include dir: ${OpenCV_INCLUDE_DIRS}")
# Defualt to ONNXRuntime CPU and use CUDA later if available
find_package(onnxruntime CONFIG REQUIRED)
find_package(flatbuffers REQUIRED)
find_package(ZeroMQ REQUIRED)

if(APPLE)
    message(STATUS "Configuring for MacBook")
    add_definitions(-DAPPLE)
elseif(UNIX AND NOT APPLE)
    message(STATUS "Configuring for Linux")
    
    find_package(CUDAToolkit)
    if(CUDAToolkit_FOUND)
        list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

        cmake_policy(SET CMP0104 NEW)
        if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
            set(CMAKE_CUDA_ARCHITECTURES 75)
        endif()


        enable_language(CUDA) 
        add_definitions(-DWITH_CUDA)
        message(STATUS "CUDA found: Compiling for Linux with GPU support")

        file(GLOB_RECURSE CUDA_SRC_FILES cuda/*.cu)
        file(GLOB_RECURSE CUDA_HDR_FILES cuda/*.h)

        file(GLOB_RECURSE CUDA_SUBDIRS LIST_DIRECTORIES true ${PROJECT_SOURCE_DIR}/cuda/*)

        foreach(SUBDIR ${CUDA_SUBDIRS})
            if(IS_DIRECTORY ${SUBDIR})
                include_directories(${SUBDIR})
            endif()
        endforeach()

        list(APPEND PLUGINS "nvinfer")
        list(APPEND PLUGINS "nvonnxparser")
        list(APPEND PLUGINS "nvparsers")

        foreach(libName ${PLUGINS})
            find_library(${libName}_lib NAMES ${libName} "/usr" PATH_SUFFIXES lib)
            list(APPEND PLUGIN_LIBS "${${libName}_lib}")
        endforeach()

        include(FindCUDA_Apriltags)
        
        find_package(ZED 4 REQUIRED)
        include_directories(
            ${CUDAToolkit_INCLUDE_DIRS}
            ${ZED_INCLUDE_DIRS}
            ${CUAPRILTAGS_INCLUDE_DIR}
        )
        link_directories(
            ${CUDAToolkit_LIBRARY_DIR}
            ${ZED_LIBRARY_DIR}
        )
        message(STATUS "CUAPRILTAGS_INCLUDE_DIR: ${CUAPRILTAGS_INCLUDE_DIR}")
        message(STATUS "CUAPRILTAGS_LIBRARY: ${CUAPRILTAGS_LIBRARY}")
    else()
        add_definitions(-DWITHOUT_CUDA)
        message(STATUS "No CUDA found: Compiling for Linux without GPU support")
    endif()
endif()

include_directories(
    ${FBS_GENERATED_DIR}
    ${OpenCV_INCLUDE_DIRS}
    ${ONNXRuntime_INCLUDE_DIRS}
    ${flatbuffers_INCLUDE_DIR}
    ${ZeroMQ_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/include
)

link_directories(
    ${OpenCV_LIBRARY_DIRS}
)

file(GLOB_RECURSE SRC_FILES src/*.cpp)
file(GLOB_RECURSE HDR_FILES include/*.h include/*.hpp)

add_executable(
    ${PROJECT_NAME} 
    ${SRC_FILES} 
    ${GENERATED_FBS_SRCS}
    ${CUDA_SRC_FILES} 
    ${CUDA_HDR_FILES}
)
add_dependencies(${PROJECT_NAME} GenerateFlatBuffers)

target_link_libraries(${PROJECT_NAME} PRIVATE
    ${OpenCV_LIBRARIES}
    ${ZeroMQ_LIBRARY}
    flatbuffers::flatbuffers
    onnxruntime::onnxruntime
)

if(UNIX AND NOT APPLE AND CUDA_FOUND)
    add_dependencies(${PROJECT_NAME} cuapriltags)
    SET(TRT_LIBS nvinfer nvonnxparser)
    target_link_libraries(${PROJECT_NAME} PRIVATE
        ${CUDA_LIBRARIES}
        ${CUDA_CUDART_LIBRARY}
        ${ZED_LIBRARIES}
        ${TRT_LIBS}
        ${CUAPRILTAGS_LIBRARY}
    )
endif()
