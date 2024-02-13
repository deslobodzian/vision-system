#!/bin/bash
set -e

TEMP_DIR=$(mktemp -d)
trap "rm -rf ${TEMP_DIR}" EXIT

cd ${TEMP_DIR}

echo "Cloning ONNX Runtime v${ONNXRUNTIME_VERSION}..."
git clone --recursive https://github.com/Microsoft/onnxruntime.git

cd onnxruntime

if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    apt-get update
    apt-get install -y cmake g++ python3-dev python3-pip
    python3 -m pip install cmake
    which cmake
    echo "Building ONNX Runtime for Linux..."
    ./build.sh --config Release --build_shared_lib --parallel --allow_running_as_root

    cd build/Linux/Release
    make install
elif [[ "$OSTYPE" == "darwin"* ]]; then
    brew install cmake python
    echo "Building ONNX Runtime for macOS..."
    #if intel cpu use x86_64
    ./build.sh --config RelWithDebInfo --build_shared_lib --parallel --compile_no_warning_as_error --skip_submodule_sync --cmake_extra_defines CMAKE_OSX_ARCHITECTURES=arm64 --allow_running_as_root

    cd build/MacOS/RelWithDebInfo
    make install
fi

echo "ONNX Runtime installed successfully."

