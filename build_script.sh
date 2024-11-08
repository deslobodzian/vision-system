#bin/bash!

echo "Building and running tests vision-system project"

CORES="$(nproc --all)"

cmake -B build -S .
cmake --build build -j$CORES

ctest --test-dir build
