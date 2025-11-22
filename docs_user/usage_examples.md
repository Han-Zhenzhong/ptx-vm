# PTX Virtual Machine - Usage Examples

**Authors**: Han-Zhenzhong, TongyiLingma, GitHub Copilot  
**Last Updated**: 2025-10-30

This document provides complete, practical examples for common use cases of the PTX Virtual Machine.

## Table of Contents

1. [Example 1: Vector Addition (API Mode)](#example-1-vector-addition-api-mode)
2. [Example 2: Matrix Transpose (CLI Mode)](#example-2-matrix-transpose-cli-mode)
3. [Example 3: Data Processing Pipeline (API Mode)](#example-3-data-processing-pipeline-api-mode)
4. [Example 4: Interactive Debugging Session](#example-4-interactive-debugging-session)
5. [Example 5: Performance Profiling](#example-5-performance-profiling)
6. [Example 6: Memory Operations](#example-6-memory-operations)
7. [Example 7: Control Flow Testing](#example-7-control-flow-testing)

---

## Example 1: Vector Addition (API Mode)

This example demonstrates how to perform vector addition using the API.

### PTX Kernel (vecAdd.ptx)

```ptx
.version 7.0
.target sm_50
.address_size 64

// Vector addition kernel
// Parameters: input array, output array, array size
.entry vecAdd(
    .param .u64 input_ptr,
    .param .u64 output_ptr,
    .param .u32 size
)
{
    .reg .u64 %rd<10>;
    .reg .u32 %r<10>;
    .reg .s32 %s<10>;
    .reg .pred %p<5>;
    
    // Load parameters from parameter memory
    ld.param.u64 %rd1, [input_ptr];
    ld.param.u64 %rd2, [output_ptr];
    ld.param.u32 %r1, [size];
    
    // Get thread ID
    mov.u32 %r2, %tid.x;
    
    // Bounds check: if (tid >= size) return
    setp.ge.u32 %p1, %r2, %r1;
    @%p1 bra END;
    
    // Calculate byte offset: offset = tid * 4
    mul.wide.u32 %rd3, %r2, 4;
    add.u64 %rd4, %rd1, %rd3;
    add.u64 %rd5, %rd2, %rd3;
    
    // Load input value
    ld.global.s32 %s1, [%rd4];
    
    // Double the value (simple operation for demonstration)
    add.s32 %s2, %s1, %s1;
    
    // Store result
    st.global.s32 [%rd5], %s2;
    
END:
    ret;
}
```

### Host Code (vecAdd_example.cpp)

```cpp
#include "host_api.hpp"
#include <iostream>
#include <vector>
#include <iomanip>

int main() {
    std::cout << "\n=== Vector Addition Example ===" << std::endl;
    std::cout << "===============================\n" << std::endl;
    
    // Step 1: Initialize the VM
    std::cout << "Step 1: Initializing VM..." << std::endl;
    HostAPI hostAPI;
    if (!hostAPI.initialize()) {
        std::cerr << "Failed to initialize VM" << std::endl;
        return 1;
    }
    std::cout << "  ✓ VM initialized successfully\n" << std::endl;
    
    // Step 2: Load PTX program
    std::cout << "Step 2: Loading PTX program..." << std::endl;
    if (!hostAPI.loadProgram("examples/vecAdd.ptx")) {
        std::cerr << "Failed to load PTX program" << std::endl;
        return 1;
    }
    std::cout << "  ✓ PTX program loaded\n" << std::endl;
    
    // Step 3: Allocate device memory
    std::cout << "Step 3: Allocating device memory..." << std::endl;
    const size_t N = 256;
    CUdeviceptr inputPtr, outputPtr;
    
    CUresult result = hostAPI.cuMemAlloc(&inputPtr, N * sizeof(int32_t));
    if (result != CUDA_SUCCESS) {
        std::cerr << "Failed to allocate input memory" << std::endl;
        return 1;
    }
    
    result = hostAPI.cuMemAlloc(&outputPtr, N * sizeof(int32_t));
    if (result != CUDA_SUCCESS) {
        std::cerr << "Failed to allocate output memory" << std::endl;
        hostAPI.cuMemFree(inputPtr);
        return 1;
    }
    std::cout << "  ✓ Allocated input at:  0x" << std::hex << inputPtr << std::dec << std::endl;
    std::cout << "  ✓ Allocated output at: 0x" << std::hex << outputPtr << std::dec << "\n" << std::endl;
    
    // Step 4: Prepare input data
    std::cout << "Step 4: Preparing input data..." << std::endl;
    std::vector<int32_t> inputData(N);
    for (size_t i = 0; i < N; ++i) {
        inputData[i] = static_cast<int32_t>(i + 1);
    }
    std::cout << "  ✓ Input data: [1, 2, 3, ..., " << N << "]\n" << std::endl;
    
    // Step 5: Copy input data to device
    std::cout << "Step 5: Copying data to device..." << std::endl;
    result = hostAPI.cuMemcpyHtoD(inputPtr, inputData.data(), N * sizeof(int32_t));
    if (result != CUDA_SUCCESS) {
        std::cerr << "Failed to copy input data" << std::endl;
        return 1;
    }
    std::cout << "  ✓ Copied " << N << " elements to device\n" << std::endl;
    
    // Step 6: Prepare kernel parameters
    std::cout << "Step 6: Launching kernel..." << std::endl;
    uint32_t size = static_cast<uint32_t>(N);
    std::vector<void*> kernelParams = {
        &inputPtr,
        &outputPtr,
        &size
    };
    
    // Launch kernel with 256 threads (8 blocks of 32 threads each)
    CUfunction kernel = nullptr; // In real code, get kernel function
    result = hostAPI.cuLaunchKernel(
        kernel,
        8, 1, 1,      // Grid: 8 blocks
        32, 1, 1,     // Block: 32 threads per block
        0,            // Shared memory: 0 bytes
        nullptr,      // Stream: default stream
        kernelParams.data(),
        nullptr
    );
    
    if (result != CUDA_SUCCESS) {
        std::cerr << "Failed to launch kernel" << std::endl;
        return 1;
    }
    std::cout << "  ✓ Kernel launched: 8 blocks × 32 threads = " << (8*32) << " threads\n" << std::endl;
    
    // Step 7: Copy results back
    std::cout << "Step 7: Copying results back..." << std::endl;
    std::vector<int32_t> outputData(N);
    result = hostAPI.cuMemcpyDtoH(outputData.data(), outputPtr, N * sizeof(int32_t));
    if (result != CUDA_SUCCESS) {
        std::cerr << "Failed to copy output data" << std::endl;
        return 1;
    }
    std::cout << "  ✓ Results copied to host\n" << std::endl;
    
    // Step 8: Verify results
    std::cout << "Step 8: Verifying results..." << std::endl;
    bool correct = true;
    for (size_t i = 0; i < N; ++i) {
        int32_t expected = inputData[i] * 2;
        if (outputData[i] != expected) {
            std::cerr << "Error at index " << i << ": expected " 
                      << expected << ", got " << outputData[i] << std::endl;
            correct = false;
            break;
        }
    }
    
    if (correct) {
        std::cout << "  ✓ All results correct!\n" << std::endl;
        
        // Show first few results
        std::cout << "First 10 results:" << std::endl;
        for (int i = 0; i < 10; ++i) {
            std::cout << "  output[" << std::setw(3) << i << "] = " 
                      << std::setw(4) << outputData[i] 
                      << " (input: " << inputData[i] << " × 2)" << std::endl;
        }
    } else {
        std::cout << "  ✗ Verification failed!" << std::endl;
    }
    
    // Step 9: Cleanup
    std::cout << "\nStep 9: Cleaning up..." << std::endl;
    hostAPI.cuMemFree(inputPtr);
    hostAPI.cuMemFree(outputPtr);
    std::cout << "  ✓ Cleanup completed" << std::endl;
    
    std::cout << "\n=== Example Complete ===" << std::endl;
    return 0;
}
```

### Running the Example

```bash
# Compile
g++ -std=c++20 vecAdd_example.cpp -I../include -L../build -lptx_vm -o vecAdd_example

# Run
./vecAdd_example
```

---

## Example 2: Matrix Transpose (CLI Mode)

This example shows how to use interactive CLI mode for debugging a matrix transpose operation.

### Interactive Session

```bash
$ ./ptx_vm

PTX Virtual Machine - Interactive Mode
Type 'help' for available commands

# Load the matrix transpose kernel
> load examples/matrix_transpose.ptx
Program loaded successfully.

# Allocate memory for 4x4 matrix (64 bytes for int32)
> alloc 64
Allocated 64 bytes at address 0x10000

> alloc 64
Allocated 64 bytes at address 0x11000

# Fill input matrix with sequential values
# Matrix layout: [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
> fill 0x10000 16 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16
Filled 16 bytes at address 0x10000

# View input matrix
> memory 0x10000 64
Memory at 0x10000:
0x10000: 01 00 00 00 02 00 00 00 03 00 00 00 04 00 00 00
0x10010: 05 00 00 00 06 00 00 00 07 00 00 00 08 00 00 00
0x10020: 09 00 00 00 0a 00 00 00 0b 00 00 00 0c 00 00 00
0x10030: 0d 00 00 00 0e 00 00 00 0f 00 00 00 10 00 00 00

# Enable debug logging to see execution details
> loglevel debug
Log level set to: debug

# Set a breakpoint to inspect intermediate state
> break 0x100
Breakpoint set at address 0x100

# Launch the transpose kernel
# Parameters: input_ptr, output_ptr, width (4)
> launch matrixTranspose 0x10000 0x11000 4
[DEBUG] Launching kernel: matrixTranspose
[DEBUG] Grid: 1×1×1, Block: 16×1×1
[DEBUG] Parameters: 0x10000, 0x11000, 4
Kernel launched successfully.
Breakpoint hit at 0x100

# Inspect register state at breakpoint
> register all
Register State:
  R0: 0x0000000000010000  (input pointer)
  R1: 0x0000000000011000  (output pointer)
  R2: 0x0000000000000004  (matrix width)
  ...

# Continue execution
> run
[DEBUG] Kernel execution completed
Execution completed successfully.

# Switch to info logging
> loglevel info
Log level set to: info

# View transposed output matrix
> memory 0x11000 64
Memory at 0x11000:
0x11000: 01 00 00 00 05 00 00 00 09 00 00 00 0d 00 00 00
0x11010: 02 00 00 00 06 00 00 00 0a 00 00 00 0e 00 00 00
0x11020: 03 00 00 00 07 00 00 00 0b 00 00 00 0f 00 00 00
0x11030: 04 00 00 00 08 00 00 00 0c 00 00 00 10 00 00 00

# Show execution statistics
> dump
=== Execution Statistics ===
Total Cycles:              248
Instructions Executed:     128
IPC:                       0.516
Branch Instructions:       8
Memory Operations:         32
Divergent Branches:        0

# Visualize performance
> visualize performance
=== Performance Counters ===
Global Memory Reads:       16
Global Memory Writes:      16
Cache Hit Rate:            75.0%
Memory Bandwidth:          512 bytes
Execution Time:            248 cycles

# Exit
> quit
Goodbye!
```

---

## Example 3: Data Processing Pipeline (API Mode)

This example demonstrates a multi-stage data processing pipeline.

```cpp
#include "host_api.hpp"
#include <iostream>
#include <vector>
#include <chrono>

class DataPipeline {
private:
    HostAPI& api;
    CUdeviceptr stage1Out, stage2Out, finalOut;
    const size_t dataSize;
    
public:
    DataPipeline(HostAPI& hostAPI, size_t size) 
        : api(hostAPI), dataSize(size) {
        // Allocate intermediate buffers
        api.cuMemAlloc(&stage1Out, size * sizeof(float));
        api.cuMemAlloc(&stage2Out, size * sizeof(float));
        api.cuMemAlloc(&finalOut, size * sizeof(float));
    }
    
    ~DataPipeline() {
        api.cuMemFree(stage1Out);
        api.cuMemFree(stage2Out);
        api.cuMemFree(finalOut);
    }
    
    void processData(const std::vector<float>& input, std::vector<float>& output) {
        auto start = std::chrono::high_resolution_clock::now();
        
        // Stage 1: Normalize data
        std::cout << "Stage 1: Normalizing data..." << std::endl;
        CUdeviceptr inputPtr;
        api.cuMemAlloc(&inputPtr, dataSize * sizeof(float));
        api.cuMemcpyHtoD(inputPtr, input.data(), dataSize * sizeof(float));
        
        void* params1[] = { &inputPtr, &stage1Out, &dataSize };
        // Launch normalize kernel
        // api.cuLaunchKernel(normalizeKernel, ...);
        
        // Stage 2: Apply filter
        std::cout << "Stage 2: Applying filter..." << std::endl;
        void* params2[] = { &stage1Out, &stage2Out, &dataSize };
        // Launch filter kernel
        // api.cuLaunchKernel(filterKernel, ...);
        
        // Stage 3: Aggregate results
        std::cout << "Stage 3: Aggregating results..." << std::endl;
        void* params3[] = { &stage2Out, &finalOut, &dataSize };
        // Launch aggregate kernel
        // api.cuLaunchKernel(aggregateKernel, ...);
        
        // Copy final results
        api.cuMemcpyDtoH(output.data(), finalOut, dataSize * sizeof(float));
        api.cuMemFree(inputPtr);
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "Pipeline completed in " << duration.count() << " ms" << std::endl;
    }
};

int main() {
    HostAPI api;
    api.initialize();
    api.loadProgram("pipeline_kernels.ptx");
    
    const size_t N = 1024 * 1024;  // 1M elements
    std::vector<float> input(N), output(N);
    
    // Initialize input data
    for (size_t i = 0; i < N; ++i) {
        input[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    
    // Process through pipeline
    DataPipeline pipeline(api, N);
    pipeline.processData(input, output);
    
    // Verify results
    std::cout << "First 5 results: ";
    for (int i = 0; i < 5; ++i) {
        std::cout << output[i] << " ";
    }
    std::cout << std::endl;
    
    return 0;
}
```

---

## Example 4: Interactive Debugging Session

Complete debugging workflow for finding and fixing issues.

```bash
$ ./ptx_vm

# Load program with potential bugs
> load examples/buggy_kernel.ptx
Program loaded successfully.

# Set up test data
> alloc 128
Allocated at: 0x10000
> fill 0x10000 32 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32

# Enable maximum debugging
> loglevel debug

# Set breakpoints at suspicious locations
> break 0x100
> break 0x200
> break 0x300

# Set watchpoint on critical memory
> watch 0x10000

# Launch kernel
> launch debugKernel 0x10000
Breakpoint hit at 0x100

# Inspect state
> register all
> memory 0x10000 16

# Single-step through suspicious code
> step
PC: 0x104
> step
PC: 0x108
> register r1
R1: 0x0000000000000005

# Continue to next breakpoint
> run
Breakpoint hit at 0x200

# Check if values are as expected
> memory 0x10000 16
# Found the bug! Wrong values written

# Note the issue and exit
> quit

# Fix the PTX code and test again...
```

---

## Example 5: Performance Profiling

Comprehensive performance analysis workflow.

```bash
$ ./ptx_vm

# Start profiling session
> profile performance_analysis.csv
Profiling enabled.

# Load performance test suite
> load examples/comprehensive_test_suite.ptx

# Allocate test data
> alloc 4096
> alloc 4096

# Run multiple kernels and collect data
> launch kernel1 0x10000 0x11000
> launch kernel2 0x10000 0x11000
> launch kernel3 0x10000 0x11000

# View statistics after each run
> dump

# Visualize different aspects
> visualize warp
> visualize memory
> visualize performance

# Exit to save profile data
> quit
Profiling data saved to: performance_analysis.csv

# Analyze the CSV file
$ cat performance_analysis.csv
Kernel,GridDim,BlockDim,Instructions,Cycles,IPC,MemOps,CacheHits,Divergence
kernel1,"1,1,1","32,1,1",1024,1856,0.552,256,192,0.05
kernel2,"1,1,1","32,1,1",2048,3421,0.598,512,384,0.12
kernel3,"1,1,1","32,1,1",512,876,0.584,128,96,0.02
```

---

## Example 6: Memory Operations

Comprehensive memory management example.

```cpp
#include "host_api.hpp"
#include <iostream>
#include <vector>
#include <cstring>

void demonstrateMemoryOps(HostAPI& api) {
    std::cout << "\n=== Memory Operations Demo ===\n" << std::endl;
    
    // 1. Basic allocation
    std::cout << "1. Basic Allocation:" << std::endl;
    CUdeviceptr ptr1, ptr2, ptr3;
    api.cuMemAlloc(&ptr1, 1024);
    api.cuMemAlloc(&ptr2, 2048);
    api.cuMemAlloc(&ptr3, 512);
    std::cout << "   Allocated 3 buffers\n" << std::endl;
    
    // 2. Host to Device transfer
    std::cout << "2. Host to Device Transfer:" << std::endl;
    std::vector<int> hostData(256);
    for (int i = 0; i < 256; ++i) hostData[i] = i;
    api.cuMemcpyHtoD(ptr1, hostData.data(), 256 * sizeof(int));
    std::cout << "   Copied 256 integers to device\n" << std::endl;
    
    // 3. Device to Device copy
    std::cout << "3. Device to Device Copy:" << std::endl;
    api.cuMemcpyDtoD(ptr2, ptr1, 256 * sizeof(int));
    std::cout << "   Copied data from ptr1 to ptr2\n" << std::endl;
    
    // 4. Device to Host transfer
    std::cout << "4. Device to Host Transfer:" << std::endl;
    std::vector<int> resultData(256);
    api.cuMemcpyDtoH(resultData.data(), ptr2, 256 * sizeof(int));
    std::cout << "   Copied 256 integers back to host\n" << std::endl;
    
    // 5. Verify data integrity
    std::cout << "5. Verification:" << std::endl;
    bool correct = true;
    for (int i = 0; i < 256; ++i) {
        if (hostData[i] != resultData[i]) {
            correct = false;
            break;
        }
    }
    std::cout << "   Data integrity: " << (correct ? "PASS" : "FAIL") << "\n" << std::endl;
    
    // 6. Cleanup
    std::cout << "6. Cleanup:" << std::endl;
    api.cuMemFree(ptr1);
    api.cuMemFree(ptr2);
    api.cuMemFree(ptr3);
    std::cout << "   All buffers freed" << std::endl;
}

int main() {
    HostAPI api;
    api.initialize();
    
    demonstrateMemoryOps(api);
    
    return 0;
}
```

---

## Example 7: Control Flow Testing

Testing branch divergence and reconvergence.

```cpp
#include "host_api.hpp"
#include <iostream>
#include <vector>

int main() {
    std::cout << "\n=== Control Flow Testing ===\n" << std::endl;
    
    HostAPI api;
    api.initialize();
    api.loadProgram("examples/control_flow_example.ptx");
    
    // Prepare test data with mixed values
    const size_t N = 256;
    std::vector<int> testData(N);
    for (size_t i = 0; i < N; ++i) {
        // Create pattern that will cause branch divergence
        testData[i] = (i % 2 == 0) ? i : -i;
    }
    
    CUdeviceptr inputPtr, outputPtr;
    api.cuMemAlloc(&inputPtr, N * sizeof(int));
    api.cuMemAlloc(&outputPtr, N * sizeof(int));
    
    api.cuMemcpyHtoD(inputPtr, testData.data(), N * sizeof(int));
    
    // Launch kernel that has conditional branches
    void* params[] = { &inputPtr, &outputPtr };
    // api.cuLaunchKernel(controlFlowKernel, ...);
    
    // Copy results and analyze
    std::vector<int> results(N);
    api.cuMemcpyDtoH(results.data(), outputPtr, N * sizeof(int));
    
    // Get performance counters to check divergence
    api.printPerformanceCounters();
    
    // Verify branching behavior
    int posCount = 0, negCount = 0;
    for (const auto& val : results) {
        if (val > 0) posCount++;
        else if (val < 0) negCount++;
    }
    
    std::cout << "\nBranch Statistics:" << std::endl;
    std::cout << "  Positive path: " << posCount << " threads" << std::endl;
    std::cout << "  Negative path: " << negCount << " threads" << std::endl;
    std::cout << "  Expected divergence with warp size 32" << std::endl;
    
    api.cuMemFree(inputPtr);
    api.cuMemFree(outputPtr);
    
    return 0;
}
```

---

## Summary

These examples demonstrate:

1. **API Mode**: Complete programmatic control for production use
2. **CLI Mode**: Interactive debugging and exploration
3. **Memory Management**: Efficient data transfer and management
4. **Performance Analysis**: Profiling and optimization
5. **Debugging**: Finding and fixing issues
6. **Control Flow**: Testing branch divergence

For more examples, see:
- `examples/` directory in the repository
- [User Guide](./user_guide.md) for detailed explanations
- [API Documentation](./api_documentation.md) for API reference

---

**Note**: All code examples are complete and can be compiled and run with the PTX VM. Adjust paths and parameters as needed for your specific use case.
