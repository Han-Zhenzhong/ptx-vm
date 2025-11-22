#!/bin/bash
# Build script for libptxrt

set -e

echo "=========================================="
echo "Building PTX Runtime Library (libptxrt)"
echo "=========================================="
echo ""

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Create build directory
BUILD_DIR="build"
if [ -d "$BUILD_DIR" ]; then
    echo "Cleaning existing build directory..."
    rm -rf "$BUILD_DIR"
fi

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Configure
echo "Configuring with CMake..."
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$SCRIPT_DIR/install"

# Build
echo ""
echo "Building..."
make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

echo ""
echo "=========================================="
echo "Build completed successfully!"
echo "=========================================="
echo ""
echo "Libraries built:"
echo "  - Static: $BUILD_DIR/libptxrt.a"
echo "  - Shared: $BUILD_DIR/libptxrt.so"
echo ""
echo "To install, run:"
echo "  cd $BUILD_DIR && make install"
echo ""
echo "To use in your CUDA programs:"
echo "  clang++ your_program.cu \\"
echo "    --cuda-path=/usr/local/cuda \\"
echo "    --cuda-gpu-arch=sm_61 \\"
echo "    -I$SCRIPT_DIR \\"
echo "    -L$BUILD_DIR \\"
echo "    -lptxrt \\"
echo "    -o your_program"
echo ""
