#!/bin/bash
set -e

OPENCV_VERSION="4.8.0"
DOWNLOAD_URL="https://github.com/opencv/opencv/archive/refs/tags/${OPENCV_VERSION}.tar.gz"

TEMP_DIR=$(mktemp -d)
trap "rm -rf ${TEMP_DIR}" EXIT

cd ${TEMP_DIR}

install_dependencies() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        apt-get update
        apt-get upgrade -y
        apt-get install -y build-essential cmake git pkg-config libgtk-3-dev \
                                libavcodec-dev libavformat-dev libswscale-dev libv4l-dev \
                                libxvidcore-dev libx264-dev libjpeg-dev libpng-dev libtiff-dev \
                                gfortran libopenblas-dev liblapack-dev libeigen3-dev
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        brew update
        brew install cmake git pkg-config gtk+3 ffmpeg \
                       gfortran openblas lapack eigen
    else
        echo "Unknown OS: $OSTYPE"
        exit 1
    fi
}

echo "Installing dependencies..."
install_dependencies

echo "Downloading OpenCV v${OPENCV_VERSION}..."
curl -L -o ${OPENCV_VERSION}.tar.gz ${DOWNLOAD_URL}

echo "Extracting..."
tar -xzf ${OPENCV_VERSION}.tar.gz

cd opencv-${OPENCV_VERSION}

echo "Building OpenCV..."
mkdir build
cd build
cmake ..
make -j$(sysctl -n hw.logicalcpu || nproc)
make install

echo "OpenCV ${OPENCV_VERSION} installed successfully."

