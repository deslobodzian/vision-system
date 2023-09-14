#!/bin/bash
set -e
APRILTAG_VERSION="3.2.0"
DOWNLOAD_URL="https://github.com/AprilRobotics/apriltag/archive/refs/tags/v${APRILTAG_VERSION}.tar.gz"

TEMP_DIR=$(mktemp -d)

cd ${TEMP_DIR}

echo "Downloading AprilTag v${APRILTAG_VERSION}..."
wget -O ${APRILTAG_VERSION}.tar.gz ${DOWNLOAD_URL}

echo "Extracting..."
tar -xzf ${APRILTAG_VERSION}.tar.gz
ls -la
cd apriltag-${APRILTAG_VERSION}

echo "Building AprilTag..."
mkdir build
cd build
cmake ..
make
sudo make install

cd -

rm -rf ${TEMP_DIR}

echo "AprilTag ${APRILTAG_VERSION} installed successfully."