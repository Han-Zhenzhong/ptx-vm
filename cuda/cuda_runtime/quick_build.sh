#!/bin/bash
# Quick build script for libptxrt

set -e

echo "==========================================="
echo "Building libptxrt (CUDA Runtime API replacement)"
echo "==========================================="
echo ""

# Navigate to cuda_runtime directory
cd "$(dirname "$0")"

# Create and enter build directory
BUILD_DIR="build"
if [ ! -d "$BUILD_DIR" ]; then
    mkdir "$BUILD_DIR"
fi

cd "$BUILD_DIR"

# Run CMake
echo "Configuring with CMake..."
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build
echo ""
echo "Building..."
make -j$(nproc 2>/dev/null || echo 4)

echo ""
echo "==========================================="
echo "Build completed successfully!"
echo "==========================================="
echo ""
echo "Libraries built:"
ls -lh libptxrt.a 2>/dev/null || echo "  Static library: build/libptxrt.a"
ls -lh libptxrt.so 2>/dev/null || echo "  Shared library: build/libptxrt.so"
echo ""
