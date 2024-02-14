include(ExternalProject)

# Set the default install path
set(CUAPRILTAGS_PREFIX "${CMAKE_BINARY_DIR}/cuapriltags" CACHE PATH "Path to CUDA AprilTags library")

ExternalProject_Add(cuapriltags
    GIT_REPOSITORY https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_nitros.git
    GIT_TAG main # Specify a specific commit or tag if needed
    PREFIX ${CUAPRILTAGS_PREFIX}
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
    LOG_DOWNLOAD ON
    UPDATE_DISCONNECTED TRUE
    GIT_CONFIG advice.detachedHead=false
    GIT_SUBMODULES "" # Add this if the project has submodules
    # Initialize and update Git LFS after cloning
    PATCH_COMMAND git lfs install && git lfs pull
)

# Determine the appropriate library directory
set(CUAPRILTAGS_LIB_DIR "")
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    set(CUAPRILTAGS_LIB_DIR "${CUAPRILTAGS_PREFIX}/src/cuapriltags/isaac_ros_nitros/lib/cuapriltags/lib_aarch64_jetpack51")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
    set(CUAPRILTAGS_LIB_DIR "${CUAPRILTAGS_PREFIX}/src/cuapriltags/isaac_ros_nitros/lib/cuapriltags/lib_x86_64_cuda_11_8")
endif()

# Set variables for parent scope
set(CUAPRILTAGS_INCLUDE_DIR "${CUAPRILTAGS_PREFIX}/src/cuapriltags/isaac_ros_nitros/lib/cuapriltags/cuapriltags" CACHE PATH "Path to cuAprilTags include directory" FORCE)
set(CUAPRILTAGS_LIBRARY "${CUAPRILTAGS_LIB_DIR}/libcuapriltags.a" CACHE FILEPATH "Path to cuAprilTags library file" FORCE)

