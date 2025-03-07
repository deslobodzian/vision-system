# Find FFmpeg components
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
    pkg_check_modules(FFMPEG_AVDEVICE QUIET libavdevice)
    pkg_check_modules(FFMPEG_AVFORMAT QUIET libavformat)
    pkg_check_modules(FFMPEG_AVCODEC QUIET libavcodec)
    pkg_check_modules(FFMPEG_SWSCALE QUIET libswscale)
    pkg_check_modules(FFMPEG_AVUTIL QUIET libavutil)
endif()

# Function to enable FFmpeg for a target
function(enable_ffmpeg_for_target target)
    # Check if all required FFmpeg components are found
    if(FFMPEG_AVDEVICE_FOUND AND FFMPEG_AVFORMAT_FOUND AND FFMPEG_AVCODEC_FOUND AND
       FFMPEG_SWSCALE_FOUND AND FFMPEG_AVUTIL_FOUND)
        # Set include directories
        target_include_directories(${target} PRIVATE
            ${FFMPEG_AVDEVICE_INCLUDE_DIRS}
            ${FFMPEG_AVFORMAT_INCLUDE_DIRS}
            ${FFMPEG_AVCODEC_INCLUDE_DIRS}
            ${FFMPEG_SWSCALE_INCLUDE_DIRS}
            ${FFMPEG_AVUTIL_INCLUDE_DIRS}
        )

        # Set link directories
        target_link_directories(${target} PRIVATE
            ${FFMPEG_AVDEVICE_LIBRARY_DIRS}
            ${FFMPEG_AVFORMAT_LIBRARY_DIRS}
            ${FFMPEG_AVCODEC_LIBRARY_DIRS}
            ${FFMPEG_SWSCALE_LIBRARY_DIRS}
            ${FFMPEG_AVUTIL_LIBRARY_DIRS}
        )

        # Link libraries
        target_link_libraries(${target} PRIVATE
            ${FFMPEG_AVDEVICE_LIBRARIES}
            ${FFMPEG_AVFORMAT_LIBRARIES}
            ${FFMPEG_AVCODEC_LIBRARIES}
            ${FFMPEG_SWSCALE_LIBRARIES}
            ${FFMPEG_AVUTIL_LIBRARIES}
        )

        # Add compile definitions
        target_compile_definitions(${target} PRIVATE FFMPEG)

        # Special handling for macOS with Homebrew
        if(APPLE)
            execute_process(
                COMMAND brew --prefix ffmpeg
                RESULT_VARIABLE BREW_FFMPEG_RESULT
                OUTPUT_VARIABLE BREW_FFMPEG_PREFIX
                ERROR_QUIET
                OUTPUT_STRIP_TRAILING_WHITESPACE
            )

            if(BREW_FFMPEG_RESULT EQUAL 0)
                message(STATUS "Found FFmpeg via Homebrew: ${BREW_FFMPEG_PREFIX}")
                target_include_directories(${target} PRIVATE "${BREW_FFMPEG_PREFIX}/include")
                target_link_directories(${target} PRIVATE "${BREW_FFMPEG_PREFIX}/lib")
            endif()
        endif()

        # Special handling for Windows with vcpkg
        if(WIN32)
            if(DEFINED ENV{VCPKG_ROOT})
                set(VCPKG_FFMPEG_DIR "$ENV{VCPKG_ROOT}/installed/x64-windows")
                if(EXISTS "${VCPKG_FFMPEG_DIR}/include/libavdevice")
                    message(STATUS "Found FFmpeg via vcpkg: ${VCPKG_FFMPEG_DIR}")
                    target_include_directories(${target} PRIVATE "${VCPKG_FFMPEG_DIR}/include")
                    target_link_directories(${target} PRIVATE "${VCPKG_FFMPEG_DIR}/lib")
                endif()
            endif()
        endif()

        message(STATUS "FFmpeg enabled for target ${target}")
        set(FFMPEG_AVAILABLE TRUE CACHE BOOL "FFmpeg availability flag" FORCE)
    else()
        message(WARNING "FFmpeg packages not found, skipping FFmpeg setup for target ${target}")
        if(PKG_CONFIG_FOUND)
            message(STATUS "Missing components:")
            if(NOT FFMPEG_AVDEVICE_FOUND)
                message(STATUS "  - libavdevice")
            endif()
            if(NOT FFMPEG_AVFORMAT_FOUND)
                message(STATUS "  - libavformat")
            endif()
            if(NOT FFMPEG_AVCODEC_FOUND)
                message(STATUS "  - libavcodec")
            endif()
            if(NOT FFMPEG_SWSCALE_FOUND)
                message(STATUS "  - libswscale")
            endif()
            if(NOT FFMPEG_AVUTIL_FOUND)
                message(STATUS "  - libavutil")
            endif()
        else()
            message(STATUS "pkg-config not found, unable to detect FFmpeg components")
        endif()
        set(FFMPEG_AVAILABLE FALSE CACHE BOOL "FFmpeg availability flag" FORCE)
    endif()
endfunction()
