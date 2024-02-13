#!/bin/bash
set -e

CPPZMQ_VERSION="4.10.0"
DOWNLOAD_URL="https://github.com/zeromq/cppzmq/archive/refs/tags/v${CPPZMQ_VERSION}.tar.gz"

TEMP_DIR=$(mktemp -d)
trap "rm -rf ${TEMP_DIR}" EXIT

cd ${TEMP_DIR}

echo "Downloading cppzmq v${CPPZMQ_VERSION}..."
curl -L -o ${CPPZMQ_VERSION}.tar.gz ${DOWNLOAD_URL}

echo "Extracting..."
tar -xzf ${CPPZMQ_VERSION}.tar.gz

cd cppzmq-${CPPZMQ_VERSION}

install_dependencies() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        apt-get update
        apt-get install -y build-essential cmake pkg-config
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        brew update
        brew install cmake pkg-config
    else
        echo "Unknown OS: $OSTYPE"
        exit 1
    fi
}

echo "Installing dependencies..."
install_dependencies

echo "Building cppzmq..."
mkdir build
cd build
cmake ..
make -j$(sysctl -n hw.logicalcpu || nproc)
make install

if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    ldconfig
fi

echo "cppzmq ${CPPZMQ_VERSION} installed successfully."

