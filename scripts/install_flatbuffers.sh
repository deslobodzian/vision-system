#!/bin/bash
set -e

FLATBUFFERS_VERSION="23.5.26"
DOWNLOAD_URL="https://github.com/google/flatbuffers/archive/refs/tags/v${FLATBUFFERS_VERSION}.tar.gz"

TEMP_DIR=$(mktemp -d)
trap "rm -rf ${TEMP_DIR}" EXIT

cd ${TEMP_DIR}

echo "Downloading FlatBuffers v${FLATBUFFERS_VERSION}..."
curl -L -o flatbuffers-${FLATBUFFERS_VERSION}.tar.gz ${DOWNLOAD_URL}

echo "Extracting..."
tar -xzf flatbuffers-${FLATBUFFERS_VERSION}.tar.gz
cd flatbuffers-${FLATBUFFERS_VERSION}

if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    apt-get update
    apt-get install -y cmake g++
elif [[ "$OSTYPE" == "darwin"* ]]; then
    brew install cmake
fi

echo "Building FlatBuffers..."
cmake -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release
make

make install

echo "FlatBuffers v${FLATBUFFERS_VERSION} installed successfully."

