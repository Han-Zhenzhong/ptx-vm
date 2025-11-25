#!/bin/bash

# Build script for CUDA programs using clang + libptxrt + PTX VM

set -e

CUDA_FILE="simple_add.cu"
OUTPUT_NAME="simple_add"
PTX_FILE="${OUTPUT_NAME}.ptx"

# Step 1: Generate PTX from CUDA source
echo "==> Step 1: Generating PTX..."
clang++ $CUDA_FILE \
  --cuda-path=/usr/local/cuda-12.6 \
  --cuda-gpu-arch=sm_70 \
  --cuda-device-only \
  -S \
  -o $PTX_FILE

echo "    PTX generated: $PTX_FILE"

# Step 2: Rebuild libptxrt
echo "==> Step 2: Rebuilding libptxrt..."
cd ..
make clean && make
cd examples

# Step 3: Compile host code and link with libptxrt + PTX VM libraries
echo "==> Step 3: Compiling host code..."
clang++ $CUDA_FILE \
  --cuda-path=/usr/local/cuda-12.6 \
  --cuda-gpu-arch=sm_70 \
  --cuda-host-only \
  -L../build \
  -lptxrt \
  -L../../../build/src/host -lhost \
  -L../../../build/src/execution -lexecution \
  -L../../../build/src/memory -lmemory \
  -L../../../build/src/registers -lregisters \
  -L../../../build/src/decoder -ldecoder \
  -L../../../build/src/parser -lparser \
  -L../../../build/src/core -lcore \
  -L../../../build/src/logger -llogger \
  -L../../../build/src/optimizer -loptimizer \
  -L../../../build/src/debugger -ldebugger \
  -o $OUTPUT_NAME

echo "    Binary compiled: $OUTPUT_NAME"

# Step 4: Set environment and run
echo "==> Step 4: Running program..."
export PTXRT_PTX_PATH=./$PTX_FILE
./$OUTPUT_NAME

echo "==> Done!"
