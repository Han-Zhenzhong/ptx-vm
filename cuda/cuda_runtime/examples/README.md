# Example CUDA Program using libptxrt

This directory contains example CUDA programs that can be compiled and linked with libptxrt.

## Prerequisites

1. Clang with CUDA support (or NVCC)
2. Built libptxrt library

## Building Examples

### Using Clang

```bash
# Compile the CUDA source
clang++ simple_add.cu \
  --cuda-path=/usr/local/cuda \
  --cuda-gpu-arch=sm_61 \
  -I../cuda/cuda_runtime \
  -L../cuda/cuda_runtime/build \
  -lptxrt \
  -o simple_add

# Run the program
./simple_add
```

### Using NVCC with custom runtime

```bash
# Compile with NVCC but link against our runtime
nvcc simple_add.cu \
  -cudart=shared \
  -I../cuda/cuda_runtime \
  -L../cuda/cuda_runtime/build \
  -lptxrt \
  -o simple_add

# Run the program
LD_LIBRARY_PATH=../cuda/cuda_runtime/build:$LD_LIBRARY_PATH ./simple_add
```

## Verifying PTX Generation

To verify that PTX is being generated and embedded:

```bash
# Generate PTX only
clang++ simple_add.cu \
  --cuda-path=/usr/local/cuda \
  --cuda-gpu-arch=sm_61 \
  --cuda-device-only \
  -S -o simple_add.ptx

# View the generated PTX
cat simple_add.ptx
```

## Expected PTX Version

For PTX 6.1 (which the VM supports), use these compute capabilities:
- `sm_61` - Compute Capability 6.1 (Pascal - GTX 1080, Tesla P40)
- `sm_60` - Compute Capability 6.0 (Pascal - Tesla P100)

## Notes

- The current implementation has empty function bodies
- Once implemented, the runtime will extract PTX from the fat binary and execute it via PTX VM
- All memory operations are simulated on the host
