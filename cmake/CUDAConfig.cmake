# Find CUDA
find_package(CUDAToolkit)

# Function to enable CUDA for a target
function(enable_cuda_for_target target)
    if(CUDAToolkit_FOUND)
        cmake_policy(SET CMP0104 NEW)
        if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
            set(CMAKE_CUDA_ARCHITECTURES 75)
        endif()

        enable_language(CUDA)
        add_definitions(-DCUDA)
        message(STATUS "CUDA found")
        target_link_libraries(${target} PRIVATE CUDA::cudart)
        target_compile_definitions(${target} PRIVATE CUDA)
        set(CUDA_AVAILABLE TRUE CACHE BOOL "CUDA availability flag" FORCE)
    else()
        message(WARNING "CUDA not found, skipping CUDA setup for target ${target}")
        set(CUDA_AVAILABLE FALSE CACHE BOOL "CUDA availability flag" FORCE)
    endif()
endfunction()
