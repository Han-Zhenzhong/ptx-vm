# User Guide for PTX Virtual Machine

**Authors**: Han-Zhenzhong, TongyiLingma, GitHub Copilot  
**Contact**: zhenzhong.han@qq.com  
**Last Updated**: 2025-10-30

## Table of Contents
- [Introduction](#introduction)
- [Getting Started](#getting-started)
- [Three Usage Modes](#three-usage-modes)
  - [Mode 1: API Programming](#mode-1-api-programming-mode)
  - [Mode 2: Interactive CLI](#mode-2-interactive-cli-mode)
  - [Mode 3: Direct Execution](#mode-3-direct-execution-mode)
- [Complete Workflow Examples](#complete-workflow-examples)
- [Command Reference](#command-reference)
- [Advanced Features](#advanced-features)
- [Troubleshooting](#troubleshooting)

## Introduction

The PTX Virtual Machine is a complete implementation for executing NVIDIA PTX (Parallel Thread Execution) intermediate code. It provides a full SIMT execution model, memory management, performance analysis, and comprehensive debugging features.

### Key Features
- ✅ Full SIMT execution model with warp scheduling
- ✅ Comprehensive parameter passing support
- ✅ Multiple usage modes: API, CLI, Direct execution
- ✅ Performance profiling and visualization
- ✅ Debugging support with breakpoints and watchpoints
- ✅ Detailed logging system with multiple levels
- ✅ Memory management with hierarchical memory model

### Prerequisites
- CMake 3.14+
- C++20 compatible compiler:
  - GCC 10+
  - Clang 12+
  - MSVC VS2019 16.10+
- Google Test (for unit testing)

### Building the Project
```bash
# Clone the repository
git clone https://gitee.com/hanzhenzhong/ptx-vm.git
cd ptx_vm

# Create build directory
mkdir build && cd build

# Configure with CMake
cmake ..

# Build the project
make
```

### Build Options
- `BUILD_TESTS=ON/OFF` - Enable/disable unit tests
- `BUILD_EXAMPLES=ON/OFF` - Enable/disable example programs
- `BUILD_DOCUMENTATION=ON/OFF` - Enable/disable documentation building

## Getting Started

### Quick Start
```bash
# Build the project
cd build
make

# Try the quickest way - run example programs
./execution_result_demo
./parameter_passing_example

# Or run PTX files directly
./ptx_vm ../examples/simple_math_example.ptx
```

## Three Usage Modes

The PTX VM provides **three different ways** to use the virtual machine, each suited for different scenarios:

| Mode | Best For | Use Case |
|------|----------|----------|
| **API Programming** | Integration | Embedding PTX-VM into your application |
| **Interactive CLI** | Debugging & Learning | Experimenting with PTX code interactively |
| **Direct Execution** | Quick Testing | Running PTX files quickly from command line |

### Mode 1: API Programming Mode

**Use this when:** You want to integrate PTX VM into your C++ application.

#### Complete API Example

```cpp
#include "host_api.hpp"
#include <iostream>
#include <vector>

int main() {
    // Step 1: Initialize the VM
    HostAPI hostAPI;
    if (!hostAPI.initialize()) {
        std::cerr << "Failed to initialize VM" << std::endl;
        return 1;
    }
    std::cout << "✓ VM initialized successfully" << std::endl;
    
    // Step 2: Load PTX program
    if (!hostAPI.loadProgram("my_kernel.ptx")) {
        std::cerr << "Failed to load PTX program" << std::endl;
        return 1;
    }
    std::cout << "✓ PTX program loaded" << std::endl;
    
    // Step 3: Allocate device memory
    const size_t dataSize = 1024;
    CUdeviceptr inputPtr, outputPtr;
    
    hostAPI.cuMemAlloc(&inputPtr, dataSize * sizeof(int));
    hostAPI.cuMemAlloc(&outputPtr, dataSize * sizeof(int));
    std::cout << "✓ Allocated device memory" << std::endl;
    
    // Step 4: Prepare and copy input data to device
    std::vector<int> inputData(dataSize);
    for (size_t i = 0; i < dataSize; ++i) {
        inputData[i] = static_cast<int>(i);
    }
    
    hostAPI.cuMemcpyHtoD(inputPtr, inputData.data(), dataSize * sizeof(int));
    std::cout << "✓ Copied input data to device" << std::endl;
    
    // Step 5: Prepare kernel parameters
    uint32_t size = static_cast<uint32_t>(dataSize);
    std::vector<void*> kernelParams = {
        &inputPtr,   // Pointer to input data
        &outputPtr,  // Pointer to output data
        &size        // Size parameter
    };
    
    // Step 6: Launch kernel
    CUfunction kernel = /* get kernel function */;
    CUresult result = hostAPI.cuLaunchKernel(
        kernel,
        1, 1, 1,      // Grid dimensions (1 block)
        32, 1, 1,     // Block dimensions (32 threads)
        0,            // Shared memory bytes
        nullptr,      // Stream
        kernelParams.data(),  // Kernel parameters
        nullptr       // Extra parameters
    );
    
    if (result == CUDA_SUCCESS) {
        std::cout << "✓ Kernel launched successfully" << std::endl;
    }
    
    // Step 7: Copy results back to host
    std::vector<int> outputData(dataSize);
    hostAPI.cuMemcpyDtoH(outputData.data(), outputPtr, dataSize * sizeof(int));
    std::cout << "✓ Copied results back to host" << std::endl;
    
    // Step 8: Display results (first 10 elements)
    std::cout << "\nResults (first 10 elements):" << std::endl;
    for (int i = 0; i < 10 && i < dataSize; ++i) {
        std::cout << "  output[" << i << "] = " << outputData[i] << std::endl;
    }
    
    // Step 9: Cleanup
    hostAPI.cuMemFree(inputPtr);
    hostAPI.cuMemFree(outputPtr);
    std::cout << "✓ Cleanup completed" << std::endl;
    
    return 0;
}
```

#### Key API Functions

**Memory Management:**
```cpp
// Allocate device memory
CUresult cuMemAlloc(CUdeviceptr* dptr, size_t bytesize);

// Free device memory
CUresult cuMemFree(CUdeviceptr dptr);

// Host to device copy
CUresult cuMemcpyHtoD(CUdeviceptr dstDevice, const void* srcHost, size_t ByteCount);

// Device to host copy
CUresult cuMemcpyDtoH(void* dstHost, CUdeviceptr srcDevice, size_t ByteCount);

// Device to device copy
CUresult cuMemcpyDtoD(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount);
```

**Program Loading:**
```cpp
// Load PTX program
bool loadProgram(const std::string& filename);

// Check if program is loaded
bool isProgramLoaded() const;
```

**Kernel Launching:**
```cpp
// Launch kernel with parameters
CUresult cuLaunchKernel(
    CUfunction f,              // Kernel function
    unsigned int gridDimX,     // Grid X dimension
    unsigned int gridDimY,     // Grid Y dimension
    unsigned int gridDimZ,     // Grid Z dimension
    unsigned int blockDimX,    // Block X dimension
    unsigned int blockDimY,    // Block Y dimension
    unsigned int blockDimZ,    // Block Z dimension
    unsigned int sharedMemBytes, // Shared memory size
    CUstream hStream,          // Stream (can be nullptr)
    void** kernelParams,       // Array of parameter pointers
    void** extra               // Extra parameters (can be nullptr)
);
```

**Debugging:**
```cpp
// Set breakpoint
bool setBreakpoint(size_t address);

// Set watchpoint
bool setWatchpoint(uint64_t address);

// Print registers
void printRegisters() const;
void printAllRegisters() const;

// Print memory
void printMemory(uint64_t address, size_t size) const;

// Print performance counters
void printPerformanceCounters() const;
```

### Mode 2: Interactive CLI Mode

**Use this when:** You want to debug, experiment, or learn PTX execution interactively.

#### Starting Interactive Mode

```bash
# Start the VM in interactive mode
./ptx_vm

# You'll see the prompt:
PTX VM> 
```

#### Complete CLI Workflow

```bash
# Step 1: Load a PTX program
> load examples/memory_ops_example.ptx
Program loaded successfully.

# Step 2: Allocate memory for input data
> alloc 1024
Allocated 1024 bytes at address 0x10000

# Step 3: Allocate memory for output data
> alloc 1024
Allocated 1024 bytes at address 0x11000

# Step 4: Fill input memory with test data
> fill 0x10000 8 1 2 3 4 5 6 7 8
Filled 8 bytes at address 0x10000

# Step 5: Launch kernel with parameters
> launch myKernel 0x10000 0x11000
Launching kernel: myKernel
Kernel launched successfully

# Step 6: View results in memory
> memory 0x11000 32
Memory at 0x11000:
0x11000: 02 04 06 08 0a 0c 0e 10 ...

# Step 7: View execution statistics
> dump
Execution Statistics:
-------------------
Total Cycles:            156
Instructions Executed:   89
IPC (Instructions/Cycle): 0.570513

# Step 8: Exit
> quit
Goodbye!
```

#### Essential CLI Commands

```bash
# Program loading
load <filename>              # Load PTX program

# Memory operations
alloc <size>                 # Allocate memory
memory <address> [size]      # View memory contents
write <address> <value>      # Write single byte
fill <addr> <count> <v1> ... # Fill multiple bytes
memcpy <dest> <src> <size>   # Copy memory
loadfile <addr> <file> <size> # Load file into memory

# Execution control
run                          # Run program
step [count]                 # Step execution
launch <kernel> [params...]  # Launch kernel with parameters

# Debugging
break <address>              # Set breakpoint
watch <address>              # Set watchpoint
register [all|predicate|pc]  # View registers

# Information
dump                         # Show execution statistics
list                         # Show disassembly
visualize <type>             # Show visualization (warp|memory|performance)

# Configuration
loglevel [level]             # Get/set log level (debug|info|warning|error)

# Exit
quit                         # Exit VM
```

### Mode 3: Direct Execution Mode

**Use this when:** You want to quickly run a PTX file from the command line.

#### Basic Usage

```bash
# Run with default log level (INFO)
./ptx_vm examples/simple_math_example.ptx

# Run with debug logging to see detailed execution
./ptx_vm --log-level debug examples/control_flow_example.ptx

# Run with minimal output (errors only)
./ptx_vm --log-level error examples/memory_ops_example.ptx
```

#### Command-Line Options

The PTX VM supports the following command-line options:

- `-h, --help` - Display help message and usage information
- `-l, --log-level LEVEL` - Set log level (debug, info, warning, error)
  - `debug`: Show detailed debug information including register values and memory operations
  - `info`: Show general information like program loading and kernel launches (default)
  - `warning`: Show warnings and errors only
  - `error`: Show errors only

**Examples**:
```bash
# Show help
./ptx_vm --help
./ptx_vm -h

# Run with different log levels
./ptx_vm --log-level debug program.ptx    # Verbose output
./ptx_vm --log-level info program.ptx     # Normal output (default)
./ptx_vm --log-level warning program.ptx  # Minimal output
./ptx_vm -l error program.ptx             # Errors only (short form)
```

## Complete Workflow Examples

### Example 1: Vector Addition (API Mode)

This example shows how to perform vector addition using the API.

**PTX Program (vecAdd.ptx):**
```ptx
.version 7.0
.target sm_50
.address_size 64

.entry vecAdd(
    .param .u64 input_ptr,
    .param .u64 output_ptr,
    .param .u32 size
)
{
    .reg .u64 %rd<10>;
    .reg .u32 %r<10>;
    .reg .s32 %s<10>;
    
    // Load parameters
    ld.param.u64 %rd1, [input_ptr];
    ld.param.u64 %rd2, [output_ptr];
    ld.param.u32 %r1, [size];
    
    // Get thread ID
    mov.u32 %r2, %tid.x;
    
    // Check bounds
    setp.ge.u32 %p1, %r2, %r1;
    @%p1 bra END;
    
    // Calculate offset
    mul.wide.u32 %rd3, %r2, 4;
    add.u64 %rd4, %rd1, %rd3;
    add.u64 %rd5, %rd2, %rd3;
    
    // Load input value
    ld.global.s32 %s1, [%rd4];
    
    // Double the value
    add.s32 %s2, %s1, %s1;
    
    // Store result
    st.global.s32 [%rd5], %s2;
    
END:
    ret;
}
```

**C++ Host Code:**
```cpp
#include "host_api.hpp"
#include <iostream>
#include <vector>

int main() {
    HostAPI hostAPI;
    hostAPI.initialize();
    
    // Allocate memory
    const size_t N = 256;
    CUdeviceptr inputPtr, outputPtr;
    hostAPI.cuMemAlloc(&inputPtr, N * sizeof(int));
    hostAPI.cuMemAlloc(&outputPtr, N * sizeof(int));
    
    // Prepare data
    std::vector<int> input(N);
    for (int i = 0; i < N; i++) {
        input[i] = i + 1;
    }
    
    // Copy to device
    hostAPI.cuMemcpyHtoD(inputPtr, input.data(), N * sizeof(int));
    
    // Load and launch
    hostAPI.loadProgram("vecAdd.ptx");
    uint32_t size = N;
    void* params[] = { &inputPtr, &outputPtr, &size };
    
    // Launch kernel
    CUfunction kernel = /* get kernel */;
    hostAPI.cuLaunchKernel(kernel, 1,1,1, 32,1,1, 0, nullptr, params, nullptr);
    
    // Copy results back
    std::vector<int> output(N);
    hostAPI.cuMemcpyDtoH(output.data(), outputPtr, N * sizeof(int));
    
    // Verify (each output should be 2x input)
    for (int i = 0; i < 10; i++) {
        std::cout << input[i] << " * 2 = " << output[i] << std::endl;
    }
    
    // Cleanup
    hostAPI.cuMemFree(inputPtr);
    hostAPI.cuMemFree(outputPtr);
    return 0;
}
```

### Example 2: Matrix Processing (CLI Mode)

This example shows interactive debugging of a matrix operation.

```bash
# Start VM
$ ./ptx_vm

# Load program
> load examples/matrix_example.ptx
Program loaded successfully.

# Allocate memory for 8x8 matrix (256 bytes)
> alloc 256
Allocated 256 bytes at address 0x10000

> alloc 256
Allocated 256 bytes at address 0x11000

# Fill input matrix with sequential values
> fill 0x10000 16 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16
Filled 16 bytes at address 0x10000

# Set debug logging to see execution details
> loglevel debug
Log level set to: debug

# Set breakpoint before kernel completes
> break 0x2000

# Launch matrix transpose kernel
> launch matrixTranspose 0x10000 0x11000 8
Launching kernel: matrixTranspose
Breakpoint hit at 0x2000

# Inspect register state
> register all
Register State:
  %r0: 0x10000
  %r1: 0x11000
  %r2: 8
  ...

# Continue execution
> run
Kernel completed successfully.

# View output matrix
> memory 0x11000 16
Memory at 0x11000:
0x11000: 01 05 09 0d 02 06 0a 0e

# Show statistics
> dump
Execution Statistics:
-------------------
Total Cycles:            324
Instructions Executed:   156

# Exit
> quit
```

### Example 3: Quick Testing (Direct Mode)

```bash
# Test a simple kernel quickly
$ ./ptx_vm examples/simple_math_example.ptx
Loading program: examples/simple_math_example.ptx
Program loaded successfully.
Executing kernel...
Kernel completed successfully.

Results:
  42 + 7 = 49
  42 - 7 = 35
  42 * 7 = 294
  42 / 7 = 6

# Test with debug output
$ ./ptx_vm --log-level debug examples/control_flow_example.ptx
[DEBUG] Loading PTX program...
[DEBUG] Parsing instruction at line 1...
[DEBUG] Instruction decoded: mov.s32 %r1, 42
...
[DEBUG] Executing mov.s32 %r1, 42
[DEBUG] Register %r1 set to 42
...
```

### Example 4: Profiling Performance

```bash
# Start VM and begin profiling
$ ./ptx_vm
> profile performance_data.csv
Profiling enabled. Output will be saved to: performance_data.csv

# Load and run program
> load examples/comprehensive_test_suite.ptx
> launch testKernel 0x10000 0x20000
> launch testKernel 0x10000 0x20000
> launch testKernel 0x10000 0x20000

# Exit to save profile
> quit
Profiling data saved to: performance_data.csv

# View profile data
$ cat performance_data.csv
Kernel,GridDim,BlockDim,Instructions,Cycles,IPC,DivergentBranches
testKernel,"1,1,1","32,1,1",256,432,0.593,12
testKernel,"1,1,1","32,1,1",256,428,0.598,11
testKernel,"1,1,1","32,1,1",256,425,0.602,10
```

### Controlling Log Output

You can control the verbosity of the VM's output using log levels:

#### In Command Line Mode
```bash
# Enable all debug output
./ptx_vm --log-level debug examples/control_flow_example.ptx

# Normal operation (default)
./ptx_vm examples/control_flow_example.ptx

# Only show errors
./ptx_vm --log-level error examples/control_flow_example.ptx
```

#### In Interactive Mode
```bash
./ptx_vm
> loglevel debug          # Enable debug logging
> loglevel info           # Set to info (default)
> loglevel warning        # Show warnings and errors only
> loglevel error          # Show errors only
> loglevel                # Display current log level
```

For more details, see the [Logging System Documentation](logging_system.md).

## Parameter Passing

### Overview
The PTX VM now supports enhanced parameter passing for kernel execution. This allows you to pass data directly to kernels, similar to how parameters are passed in real CUDA applications.

### Memory Management Commands

#### alloc
Allocate memory in the VM:
```bash
> alloc <size>
```
Example:
```bash
> alloc 1024
Allocated 1024 bytes at address 0x10000
```

#### memcpy
Copy memory within the VM:
```bash
> memcpy <dest> <src> <size>
```
Example:
```bash
> memcpy 0x20000 0x10000 256
Copied 256 bytes from 0x10000 to 0x20000
```

#### write
Write a single byte value to memory:
```bash
> write <address> <value>
```
Example:
```bash
> write 0x10000 42
Wrote value 42 to address 0x10000
```

#### fill
Fill memory with multiple values:
```bash
> fill <address> <count> <value1> [value2] ...
```
Example:
```bash
> fill 0x10000 4 1 2 3 4
Filled 4 bytes at address 0x10000
```

#### loadfile
Load data from a file into VM memory:
```bash
> loadfile <address> <file> <size>
```
Example:
```bash
> loadfile 0x10000 data.bin 1024
Loaded 1024 bytes from data.bin to address 0x10000
```

### Kernel Launching with Parameters

#### launch
Launch a kernel with parameters:
```bash
> launch <kernel_name> [param1] [param2] ...
```
Each parameter should be a memory address where the parameter data is stored.
Example:
```bash
> launch vectorAdd 0x10000 0x20000
Launching kernel: vectorAdd
Kernel launched successfully
```

### Complete Parameter Passing Workflow

1. **Load a PTX program**:
   ```bash
   > load examples/memory_ops_example.ptx
   ```

2. **Allocate memory for parameters**:
   ```bash
   > alloc 4096
   Allocated 4096 bytes at address 0x10000
   > alloc 4096
   Allocated 4096 bytes at address 0x10100
   ```

3. **Prepare parameter data**:
   ```bash
   # Write specific values
   > fill 0x10000 4 1 2 3 4
   
   # Or load data from a file
   > loadfile 0x10100 input.bin 1024
   ```

4. **Launch the kernel with parameters**:
   ```bash
   > launch myKernel 0x10000 0x10100
   Launching kernel: myKernel
   Kernel launched successfully
   ```

5. **Examine results**:
   ```bash
   > memory 0x10100 256
   ```

### Example with Real Code
Here's a complete example of using the parameter passing mechanism:

```cpp
#include "host_api.hpp"
#include <vector>

int main() {
    HostAPI hostAPI;
    hostAPI.initialize();
    
    // Allocate memory for input and output data
    CUdeviceptr inputPtr, outputPtr;
    hostAPI.cuMemAlloc(&inputPtr, 1024 * sizeof(int));
    hostAPI.cuMemAlloc(&outputPtr, 1024 * sizeof(int));
    
    // Prepare input data
    std::vector<int> inputData(1024);
    for (int i = 0; i < 1024; i++) {
        inputData[i] = i;
    }
    
    // Copy input data to VM
    hostAPI.cuMemcpyHtoD(inputPtr, inputData.data(), 1024 * sizeof(int));
    
    // Launch kernel with parameters
    std::vector<void*> kernelParams = {
        reinterpret_cast<void*>(inputPtr),
        reinterpret_cast<void*>(outputPtr),
        nullptr  // Null terminator
    };
    
    hostAPI.cuLaunchKernel(
        functionHandle,  // Kernel function handle
        1, 1, 1,         // Grid dimensions
        32, 1, 1,        // Block dimensions
        0,               // Shared memory size
        nullptr,         // Stream
        kernelParams.data(),  // Kernel parameters
        nullptr          // Extra parameters
    );
    
    // Copy results back
    std::vector<int> outputData(1024);
    hostAPI.cuMemcpyDtoH(outputData.data(), outputPtr, 1024 * sizeof(int));
    
    // Clean up
    hostAPI.cuMemFree(inputPtr);
    hostAPI.cuMemFree(outputPtr);
    
    return 0;
}
```

## Understanding PTX Execution Results

### Overview
When PTX virtual instructions are executed, the results are stored in various locations depending on the program:

1. **Memory**: Most results are stored in global memory at addresses specified by the program
2. **Registers**: Intermediate values are stored in registers during execution
3. **Performance counters**: Execution statistics are collected in performance counters

### Memory Results
PTX programs typically store their final results in global memory. The exact location depends on the program's parameters:

```ptx
// Example from simple_math_example.ptx
st.global.s32 [%r0], %r3;      // Store add result at memory address in %r0
st.global.s32 [%r0+4], %r4;    // Store subtract result at memory address + 4
```

To examine memory results:
1. Use the `memory` command in CLI mode
2. Copy data back to host using `cuMemcpyDtoH` in API mode

### Performance Statistics
The VM collects various performance statistics during execution:

```bash
> dump
Execution Statistics:
-------------------
Total Cycles:            42
Instructions Executed:   25
IPC (Instructions/Cycle): 0.595238
Register Utilization:    18.75%
Spill Operations:        0
Global Memory Reads:     3
Global Memory Writes:    5
Shared Memory Reads:     0
Shared Memory Writes:    0
Local Memory Reads:      0
Local Memory Writes:     0
Branches:                1
Divergent Branches:      0
Warp Switches:           0
TLB Misses:              0
Page Faults:             0
Cache Hit Rate:          0
```

### Execution Flow
1. **Program Loading**: PTX program is parsed and decoded into internal representation
2. **Memory Setup**: Input data is prepared in VM memory
3. **Execution**: Instructions are executed in order (with branching)
4. **Result Storage**: Final results are stored in designated memory locations
5. **Statistics Collection**: Performance data is gathered throughout execution

### Example Execution Walkthrough
Let's walk through the execution of `simple_math_example.ptx`:

1. **Load Program**:
   ```bash
   > load examples/simple_math_example.ptx
   Program loaded successfully.
   ```

2. **Allocate Memory for Results**:
   ```bash
   > alloc 1024
   Allocated 1024 bytes at address 0x10000
   ```

3. **Execute Program**:
   ```bash
   > run
   Starting program execution...
   Program completed successfully.
   ```

4. **View Results**:
   ```bash
   > memory 0x10000 20
   Memory at 0x10000:
   0x10000: 31 00 00 00 23 00 00 00 26 01 00 00 06 00 00 00
   0x10010: 00 00 00 00
   ```

5. **View Performance Statistics**:
   ```bash
   > dump
   Execution Statistics:
   -------------------
   Total Cycles:            42
   Instructions Executed:   25
   IPC (Instructions/Cycle): 0.595238
   ```

### Interpreting Results
In the above example, the memory contains the results of arithmetic operations:
- 0x00000031 (49 in decimal) = 42 + 7 (addition result)
- 0x00000023 (35 in decimal) = 42 - 7 (subtraction result)
- 0x00000126 (294 in decimal) = 42 * 7 (multiplication result)
- 0x00000006 (6 in decimal) = 42 / 7 (division result)
- 0x00000000 (0 in decimal) = 42 % 7 (remainder result)

These values are stored as little-endian 32-bit integers in consecutive memory locations.

## Example Programs
The project includes several example PTX programs in the `examples/` directory:
- `simple_math_example.ptx` - Basic mathematical operations
- `control_flow_example.ptx` - Branches and control flow
- `memory_ops_example.ptx` - Memory operations
- `divergence_handling_examples.md` - Examples for divergence handling
- `memory_optimization_examples.md` - Examples for memory optimization

## Command Reference

### visualize
Display visualization of execution state.

```bash
> visualize <type>
```

Where `<type>` can be:
- `warp` - Warp execution visualization
- `memory` - Memory access visualization
- `performance` - Performance counter display

Examples:
```bash
> visualize warp        # Show warp execution state
> visualize memory      # Show memory access patterns
> visualize performance # Show performance counters
```

The VM provides a comprehensive command-line interface for interacting with the virtual machine.

### load
Load a PTX or CUDA binary file into the VM.
```bash
> load <filename>
```

### run
Execute the loaded program.
```bash
> run
```

### step
Execute one instruction at a time.
```bash
> step [number_of_instructions]
```

### break
Set a breakpoint at a specific address.
```bash
> break <address>
```

### watch
Set a watchpoint at a specific memory address.
```bash
> watch <address>
```

### register
Display register information.
```bash
> register [all|predicate|pc]
```

### memory
Display memory contents.
```bash
> memory <address> [size]
```

### alloc
Allocate memory in the VM.
```bash
> alloc <size>
```

### memcpy
Copy memory within the VM.
```bash
> memcpy <dest> <src> <size>
```

### write
Write a single byte value to a specific memory address.
```bash
> write <address> <value>
```
Value must be between 0-255 (one byte). This command is useful for initializing memory with specific values.
Example:
```bash
> write 0x10000 42
Wrote value 42 to address 0x10000
```

### fill
Fill memory with multiple byte values starting at a specific address.
```bash
> fill <address> <count> <value1> [value2] ...
```
This command writes multiple byte values to consecutive memory locations. Count specifies how many values to write, and each value must be between 0-255.
Example:
```bash
> fill 0x10000 4 1 2 3 4
Filled 4 bytes at address 0x10000
```

### loadfile
Load data from a file into VM memory at a specific address.
```bash
> loadfile <address> <file> <size>
```
This command reads data from a file and writes it to VM memory. It's useful for loading larger datasets or binary files into the VM.
Example:
```bash
> loadfile 0x10000 data.bin 1024
Loaded 1024 bytes from data.bin to address 0x10000
```

### launch
Launch a kernel with parameters.
```bash
> launch <kernel_name> [param1] [param2] ...
```
Each parameter should be a memory address where the parameter data is stored.
Example:
```bash
> launch vectorAdd 0x10000 0x20000
Launching kernel: vectorAdd
Kernel launched successfully
```

### profile
Start profiling session.
```bash
> profile <output_file.csv>
```

### dump
Output execution statistics.
```bash
> dump
```

### list
List loaded program disassembly.
```bash
> list
```

### quit
Exit the virtual machine.
```bash
> quit
```

## Advanced Features

### Debugging
The VM provides comprehensive debugging capabilities:
- Breakpoints
- Watchpoints
- Step-by-step execution
- Register inspection
- Memory inspection
- Visualization tools

#### Setting Breakpoints
```bash
> break 0x1000  # Set breakpoint at address 0x1000
> break 0x2000  # Set another breakpoint
> run            # Run until breakpoint
```

#### Using Watchpoints
```bash
> watch 0x4000  # Set watchpoint at address 0x4000
> run            # Run until watchpoint triggered
```

#### Using Visualization
```bash
> load examples/control_flow_example.ptx
> run
> visualize warp        # View warp execution
> visualize memory      # View memory access patterns
> visualize performance # View performance counters
```

### Profiling
The VM can collect detailed performance metrics:
```bash
> profile performance.csv  # Start profiling
> run                     # Run program
> quit                    # Exit and save profile
```

The profile output includes:
- Instructions executed
- Execution time
- IPC (instructions per cycle)
- Divergence statistics
- Memory access patterns
- Cache performance
- Bank conflicts
- TLB performance

### Execution Statistics
The VM can output detailed execution statistics:
```bash
> dump
```

Statistics include:
- Instruction count (total, by type)
- Execution time in cycles
- Divergence metrics (paths, depth, rate)
- Memory access statistics (global, shared, local)
- Cache hit/miss rates
- Bank conflicts
- Register usage
- TLB performance

### Disassembly
The VM can display the loaded program's disassembly:
```bash
> list
```

## Troubleshooting

### Common Issues

#### Invalid Program
If you encounter an error loading a PTX program, ensure that:
- The PTX file is valid
- The file path is correct
- The file is not empty
- The file contains a valid PTX program

#### Breakpoint Not Hit
If breakpoints are not being hit, check that:
- The breakpoint address is valid
- The instruction at the address is executable
- The program counter reaches the address

#### Memory Access Errors
For memory access issues:
- Verify the address is valid
- Check memory permissions
- Ensure memory is allocated
- Validate access size

#### Divergence Handling Issues
For divergence-related problems:
- Check for proper reconvergence points
- Verify divergence algorithm
- Monitor divergence depth
- Use performance counters to track divergence impact

#### Visualization Issues
If visualization commands are not working:
- Ensure a program has been loaded and executed
- Check that the visualization type is valid (warp, memory, performance)
- Verify that the VM has collected the required data

### Error Messages
Common error messages and their solutions:

| Error Message | Cause | Solution |
|--------------|------|----------|
| "Failed to load program: INVALID_PROGRAM" | Invalid PTX format | Validate PTX syntax |
| "Breakpoint set failed" | Invalid address | Check address validity |
| "Invalid memory access" | Invalid address | Check memory allocation |
| "Execution failed" | Internal error | Check logs for details |
| "Cache miss rate too high" | Poor memory access pattern | Optimize memory access |
| "Divergence overhead" | High branch divergence | Optimize control flow |
| "No program loaded" | Visualization without loaded program | Load and run a program first |

### Debugging Tips
- Use the `dump` command to get execution statistics
- Use `step` to execute instructions one at a time
- Use `register` to inspect register state
- Use `memory` to inspect memory contents
- Use `list` to see program disassembly
- Use `visualize` to get detailed runtime information
- Enable verbose logging for detailed execution traces

## Performance Optimization

### Best Practices
- Use coalesced memory access patterns
- Minimize branch divergence
- Optimize register usage
- Use shared memory for block-level communication
- Minimize memory bank conflicts
- Optimize cache usage
- Use profiling to identify bottlenecks

### Performance Metrics
Key metrics to monitor:
- IPC (instructions per cycle)
- Divergence rate
- Cache hit rate
- Memory bandwidth
- Bank conflicts
- Execution time
- TLB hit rate

### Optimization Techniques
- Use memory coalescing
- Optimize warp scheduling
- Minimize divergence depth
- Use efficient memory access patterns
- Optimize register allocation
- Use caching effectively
- Minimize memory bank conflicts

### Using Visualization for Optimization
The visualization features can help with optimization:
- Use `visualize warp` to identify divergence issues
- Use `visualize memory` to analyze memory access patterns
- Use `visualize performance` to identify performance bottlenecks

## Technical Reference

### Supported PTX Features
The VM supports most PTX version 7.0 features including:
- Math operations
- Control flow instructions
- Memory operations
- Predicate operations
- Warp-level primitives
- Shared memory
- Special registers
- Synchronization primitives (warp, CTA, grid, memory barriers)

### Execution Model
The VM implements a SIMT execution model with:
- Warp size: 32 threads
- Maximum threads per block: 1024
- Shared memory size: 48KB
- Register file size: 64 registers per thread

### Memory Model
The VM implements a hierarchical memory model with:
- Global memory: 4GB
- Shared memory: 48KB
- Constant memory: 64KB
- Texture memory: 64KB
- Register file: 64 registers per thread

### Divergence Handling
The VM supports multiple divergence handling algorithms:
- Basic reconvergence
- CFG-based reconvergence
- Stack-based predication

### Performance Counters
Available performance counters:
- Instructions executed (total, by type)
- Cycles
- Divergent branches
- Cache hits/misses
- Memory accesses (global, shared, local)
- Bank conflicts
- TLB hits/misses
- Reconvergence events

## Examples

### Basic Math Example
```bash
> load examples/simple_math_example.ptx
> run
> dump
```

### Control Flow Example
```bash
> load examples/control_flow_example.ptx
> step 10
> register pc
> dump
```

### Memory Operations Example
```bash
> load examples/memory_ops_example.ptx
> run
> memory 0x1000
> dump
```

### Parameter Passing Example
```bash
> load examples/memory_ops_example.ptx
> alloc 1024
Allocated 1024 bytes at address 0x10000
> alloc 1024
Allocated 1024 bytes at address 0x10100
> fill 0x10000 8 1 2 3 4 5 6 7 8
Filled 8 bytes at address 0x10000
> loadfile 0x10100 data.bin 256
Loaded 256 bytes from data.bin to address 0x10100
> launch myKernel 0x10000 0x10100
Launching kernel: myKernel
Kernel launched successfully
> memory 0x10100 32
```

### Visualization Example
```bash
> load examples/control_flow_example.ptx
> run
> visualize warp
> visualize memory
> visualize performance
```

### Profiling Example
```bash
> profile output.csv
> load examples/control_flow_example.ptx
> run
> quit
```

This will create a profile output file with detailed performance metrics.

## Future Improvements
Planned enhancements for the VM include:
- GUI interface for visualization
- Enhanced debugging features
- Better performance analysis tools
- Support for newer PTX versions
- Enhanced memory system
- Better divergence handling
- Enhanced documentation
- More comprehensive examples
- Improved error handling
- Enhanced logging and tracing