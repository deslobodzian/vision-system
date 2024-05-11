#!/bin/bash

find . \( -name '*.cpp' -or -name '*.hpp' -or -name '*.cu' -or -name '*.h' \) -exec clang-format -i {} +

echo "All relevant files have been formatted!"

