cmake_minimum_required(VERSION 3.22...4.0)

include(FetchContent)

# Catch2
FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/catch2.git
    GIT_TAG v3.10.0
    FIND_PACKAGE_ARGS
)
FetchContent_MakeAvailable(Catch2)

set(CATCH2_EXTRAS_DIR "${catch2_SOURCE_DIR}/extras" CACHE PATH "Catch2 extras")

# FetchContent_Declare(
#    wpilib
#    GIT_REPOSITORY https://github.com/wpilibsuite/allwpilib.git
#GIT_TAG v2025.3.2
#modules)

