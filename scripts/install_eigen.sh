#!/bin/bash
set -e

EIGEN_VERSION="3.4.0"
DOWNLOAD_URL="https://gitlab.com/libeigen/eigen/-/archive/${EIGEN_VERSION}/eigen-${EIGEN_VERSION}.tar.gz"

TEMP_DIR=$(mktemp -d)

cd ${TEMP_DIR}

echo "Downloading Eigen v${EIGEN_VERSION}..."
wget -O ${EIGEN_VERSION}.tar.gz ${DOWNLOAD_URL}

echo "Extracting..."
tar -xzf ${EIGEN_VERSION}.tar.gz

cd eigen-${EIGEN_VERSION}

echo "Building Eigen..."
mkdir build_eigen
cd build_eigen
cmake ..
make -j$(nproc)
sudo make install

cd -

rm -rf ${TEMP_DIR}

echo "Eigen ${EIGEN_VERSION} installed successfully."