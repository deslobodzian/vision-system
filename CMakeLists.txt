cmake_minimum_required(VERSION 3.5)
project(Vision_System)

# Set the C++ standard to C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

# Compiler options
add_compile_options(-pthread -O3)

# Find OpenCV package for all platforms
find_package(OpenCV REQUIRED)

# Define platform-specific preprocessor flags
if(APPLE)
    add_definitions(-DAPPLE)
    message(STATUS "Configuring for MacBook")
elseif(UNIX AND NOT APPLE)
    message(STATUS "Configuring for Linux")
    
    # Check for CUDA using the language support
    enable_language(CUDA)
    find_package(CUDA)
    if(CUDA_FOUND)
        add_definitions(-DLINUX_WITH_CUDA)
        message(STATUS "CUDA found: Compiling for Linux with GPU support")
        
        # Find and configure ZED SDK if CUDA is found
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

# Include header files
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/include
)

# Define source and header files for the target
file(GLOB_RECURSE SRC_FILES src/*.cpp)
file(GLOB_RECURSE HDR_FILES include/*.h include/*.hpp)

# Add executable target
add_executable(${PROJECT_NAME} ${SRC_FILES})

# Configure target dependencies
target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenCV_LIBRARIES})

if(UNIX AND NOT APPLE AND CUDA_FOUND)
    target_link_libraries(${PROJECT_NAME} PRIVATE
        ${CUDA_LIBRARIES} ${CUDA_CUDART_LIBRARY}
        ${ZED_LIBRARIES}
    )
endif()