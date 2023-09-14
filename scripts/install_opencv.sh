#!/bin/bash
set -e

OPENCV_VERSION="4.8.0"
DOWNLOAD_URL="https://github.com/opencv/opencv/archive/refs/tags/${OPENCV_VERSION}.tar.gz"

TEMP_DIR=$(mktemp -d)

cd ${TEMP_DIR}

echo "Updating and upgrading system packages..."
sudo apt-get update
sudo apt-get upgrade -y

echo "Installing dependencies..."
sudo apt-get install -y build-essential cmake git pkg-config libgtk-3-dev \
                         libavcodec-dev libavformat-dev libswscale-dev libv4l-dev \
                         libxvidcore-dev libx264-dev libjpeg-dev libpng-dev libtiff-dev \
                         gfortran libatlas-base-dev python3-dev

echo "Downloading OpenCV v${OPENCV_VERSION}..."
wget -O ${OPENCV_VERSION}.tar.gz ${DOWNLOAD_URL}

echo "Extracting..."
tar -xzf ${OPENCV_VERSION}.tar.gz

cd opencv-${OPENCV_VERSION}

echo "Building OpenCV..."
mkdir build
cd build
cmake ..
make -j$(nproc)
sudo make install

cd -

rm -rf ${TEMP_DIR}

echo "OpenCV ${OPENCV_VERSION} installed successfully."