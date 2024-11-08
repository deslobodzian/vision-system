#!/bin/bash

find . \( -name '*.cc' -or -name '*.h' -or -name '*.cu' \) -exec clang-format -i {} +

echo "All relevant files have been formatted!"

