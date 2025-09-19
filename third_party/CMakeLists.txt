cmake_minimum_required(VERSION 3.22...4.0)

include(FetchContent)

# Catch2
FetchContent_Declare(
    catch2
    GIT_REPOSITORY https://github.com/catchorg/catch2.git
    GIT_TAG v3.8.1
)
FetchContent_MakeAvailable(catch2)

# Make Catch2's CTest integration (extras/Catch.cmake) visible to siblings
# by exporting its path via a cache var the parent can read.
FetchContent_GetProperties(catch2)
set(CATCH2_EXTRAS_DIR "${catch2_SOURCE_DIR}/extras" CACHE PATH "Catch2 extras")

