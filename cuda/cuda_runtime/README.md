# PTX Runtime Library (libptxrt)

This directory contains a drop-in replacement for NVIDIA's CUDA Runtime API library (`libcudart`).

## Overview

The PTX Runtime Library provides the same API as NVIDIA's CUDA Runtime, allowing standard CUDA programs to run on the PTX virtual machine without modification.

## Features

- **Memory Management**: `cudaMalloc`, `cudaFree`, `cudaMemcpy`, `cudaMemset`
- **Device Management**: `cudaDeviceSynchronize`, `cudaGetDeviceCount`, `cudaSetDevice`
- **Kernel Launch**: `cudaLaunchKernel`, `cudaConfigureCall`, `cudaLaunch`
- **Registration**: `__cudaRegisterFatBinary`, `__cudaRegisterFunction`
- **Stream Management**: `cudaStreamCreate`, `cudaStreamDestroy`, `cudaStreamSynchronize`
- **Event Management**: `cudaEventCreate`, `cudaEventRecord`, `cudaEventElapsedTime`
- **Error Handling**: `cudaGetLastError`, `cudaGetErrorString`

## Building

```bash
cd cuda/cuda_runtime
mkdir build && cd build
cmake ..
make
```

This will create:
- `libptxrt.a` - Static library
- `libptxrt.so` - Shared library

## Usage

### Compiling CUDA code with Clang

```bash
clang++ your_program.cu \
  --cuda-path=/usr/local/cuda \
  --cuda-gpu-arch=sm_80 \
  -O2 -L/path/to/libptxrt -lptxrt \
  -o your_program
```

### Compiling CUDA code with NVCC (using our runtime)

```bash
nvcc your_program.cu \
  -cudart=shared \
  -L/path/to/libptxrt -lptxrt \
  -o your_program
```

## Implementation Status

Currently, all functions are declared but have **empty implementations** marked with TODO comments. The implementations need to:

1. **Fat Binary Registration** (`__cudaRegisterFatBinary`):
   - Parse the `.nv_fatbin` ELF section
   - Extract PTX code
   - Store PTX for kernel launch

2. **Function Registration** (`__cudaRegisterFunction`):
   - Map host function pointers to kernel names
   - Build lookup table for kernel dispatch

3. **Memory Management**:
   - Allocate simulated device memory (can use malloc)
   - Track allocations for proper cleanup

4. **Kernel Launch** (`cudaLaunchKernel`):
   - Look up kernel PTX by function pointer
   - Marshal arguments
   - Invoke PTX VM with grid/block configuration

5. **Synchronization**:
   - Wait for PTX VM execution to complete

## Integration with PTX VM

The library needs to link with the PTX VM and call its API to execute kernels. The typical flow is:

```
User CUDA Code
     ↓
cudaLaunchKernel()
     ↓
Look up kernel PTX
     ↓
ptx_vm_execute(ptx, grid, block, args)
     ↓
PTX VM interprets and executes
```

## Files

- `cuda_runtime.h` - Header with all API declarations
- `cuda_runtime.cpp` - Implementation (currently stubs)
- `CMakeLists.txt` - Build configuration
- `README.md` - This file

## Next Steps

1. Implement fat binary parsing to extract PTX
2. Implement kernel registration and lookup
3. Implement memory management with host memory simulation
4. Connect kernel launch to PTX VM execution
5. Add proper error handling and reporting
6. Test with sample CUDA programs

## References

- CUDA Runtime API Documentation: https://docs.nvidia.com/cuda/cuda-runtime-api/
- PTX ISA Documentation: https://docs.nvidia.com/cuda/parallel-thread-execution/
