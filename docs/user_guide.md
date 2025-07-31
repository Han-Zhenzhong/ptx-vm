# User Guide for PTX Virtual Machine

## Introduction
This document provides essential information for end-users and application developers working with the PTX Virtual Machine. It covers installation, basic usage, command reference, and troubleshooting.

**Author**: Zhenzhong Han <zhenzhong.han@qq.com>

## Installation

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

## Basic Usage

### Running the VM
The VM can be run in two modes: direct execution or interactive CLI mode.

#### Direct Execution
```bash
./ptx_vm examples/simple_math_example.ptx
```

#### Interactive CLI Mode
```bash
./ptx_vm
> load examples/control_flow_example.ptx
> run
> dump
> exit
```

### Example Programs
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