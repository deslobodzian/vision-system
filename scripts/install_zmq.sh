#!/bin/bash
set -e

ZEROMQ_VERSION="4.3.4"
DOWNLOAD_URL="https://github.com/zeromq/libzmq/archive/refs/tags/v${ZEROMQ_VERSION}.tar.gz"

TEMP_DIR=$(mktemp -d)

cd ${TEMP_DIR}

echo "Downloading ZeroMQ v${ZEROMQ_VERSION}..."
wget -O ${ZEROMQ_VERSION}.tar.gz ${DOWNLOAD_URL}

echo "Extracting..."
tar -xzf ${ZEROMQ_VERSION}.tar.gz

cd libzmq-${ZEROMQ_VERSION}

echo "Installing dependencies..."
sudo apt-get update
sudo apt-get install -y build-essential autoconf libtool pkg-config

echo "Building ZeroMQ..."
mkdir build
cd build
cmake ..
make -j$(nproc)
sudo make install
sudo ldconfig

cd -

rm -rf ${TEMP_DIR}

echo "ZeroMQ ${ZEROMQ_VERSION} installed successfully."