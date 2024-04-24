#!/bin/bash
set -e

WPILIB_VERSION="2024.3.2"
DOWNLOAD_URL="https://github.com/wpilibsuite/allwpilib/archive/refs/tags/v${WPILIB_VERSION}.tar.gz"

TEMP_DIR=$(mktemp -d)
trap '{
    if [[ $? -ne 0 ]]; then
        echo "Installation failed, keeping the files for inspection."
        echo "Files retained at: ${TEMP_DIR}"
    else
        echo "Installation complete. Cleaning up..."
        rm -rf "${TEMP_DIR}"
    fi
}' EXIT

cd "${TEMP_DIR}"

echo "Downloading WPILib v${WPILIB_VERSION}..."
curl -L -o "${WPILIB_VERSION}.tar.gz" "${DOWNLOAD_URL}"

echo "Extracting WPILib..."
tar -xzf "${WPILIB_VERSION}.tar.gz"

cd "allwpilib-${WPILIB_VERSION}"

install_dependencies() {
    echo "Installing dependencies..."
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        sudo apt-get update
        sudo apt-get install -y cmake git build-essential
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        brew update
        brew install cmake git
    else
        echo "Unknown OS: $OSTYPE"
        exit 1
    fi
}

install_dependencies

echo "Building and configuring CMake for WPILib..."
mkdir build_wpi
cd build_wpi

cmake .. -DWITH_JAVA=OFF -DWITH_SHARED_LIBS=OFF -DWITH_CSCORE=OFF -DWITH_NTCORE=ON -DWITH_WPIMATH=ON -DWITH_WPILIB=ON -DWITH_EXAMPLES=OFF -DWITH_TESTS=OFF -DWITH_GUI=OFF -DWITH_SIMULATION_MODULES=OFF

echo "Compiling WPILib..."
make -j$(nproc)

echo "Installing WPILib..."
sudo make install

if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    sudo ldconfig
fi

