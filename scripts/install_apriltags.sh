#!/bin/bash
set -e

APRILTAG_VERSION="3.2.0"
DOWNLOAD_URL="https://github.com/AprilRobotics/apriltag/archive/refs/tags/v${APRILTAG_VERSION}.tar.gz"

TEMP_DIR=$(mktemp -d)
trap "rm -rf ${TEMP_DIR}" EXIT

cd ${TEMP_DIR}

echo "Downloading AprilTag v${APRILTAG_VERSION}..."
curl -L -o ${APRILTAG_VERSION}.tar.gz ${DOWNLOAD_URL}

echo "Extracting..."
tar -xzf ${APRILTAG_VERSION}.tar.gz
cd apriltag-${APRILTAG_VERSION}

echo "Building AprilTag..."
mkdir build
cd build
cmake ..
make

OS=$(uname -s)
case "$OS" in
    Linux*)     
        make install;;
    Darwin*)    
        cp -R ../ /usr/local/Cellar/apriltag/${APRILTAG_VERSION}
        brew link apriltag;;
    *)          
        echo "Unknown operating system: $OS"
        exit 1;;
esac

echo "AprilTag ${APRILTAG_VERSION} installed successfully."

