# NVIDIA PTX Virtual Machine

A virtual machine implementation for executing NVIDIA PTX (Parallel Thread Execution) intermediate code. This project provides a complete VM architecture with advanced execution features, memory optimizations, and comprehensive tooling.

## Features

### Core Execution Engine
- Full SIMT (Single Instruction Multiple Threads) execution model
- Warp scheduling with dynamic thread mask management
- Predicated execution support for conditional operations
- Comprehensive divergence handling with multiple reconvergence algorithms
- Performance counters for detailed execution statistics

### Memory System
- Hierarchical memory model with separate spaces
- Data cache simulation with configurable parameters
- Shared memory bank conflict detection
- Memory access pattern analysis and optimization
- TLB and page fault handling for virtual memory

### Optimization Features
- Dynamic register allocation framework
- Instruction scheduling optimizations
- Memory coalescing optimizations
- Cache configuration flexibility

### Integration Layer
- Host API design for easy integration
- CLI interface for manual execution and debugging
- CUDA binary loading infrastructure
- Enhanced debugging interface with watchpoints

### Testing and Validation
- Comprehensive unit test suite
- Integration tests for system-level behavior
- Performance benchmarks
- Example programs for demonstration

### Documentation
- Developer guide with code structure overview
- User guide with command reference
- API documentation for developers
- Technical documentation for key components
- Performance testing framework
- Warp scheduler implementation details
- Predicate handler implementation details
- Divergence handling implementation details
- Memory optimizations implementation details
- CUDA binary loader implementation details
- Memory optimization examples
- Divergence handling performance testing
- Visualization features

## Building the Project

### Prerequisites
- CMake 3.14+
- C++20 compatible compiler:
  - GCC 10+
  - Clang 12+
  - MSVC VS2019 16.10+
- Google Test (for unit testing)

### Build Instructions
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

## Usage

### Basic Execution
```bash
./ptx_vm examples/simple_math_example.ptx
```

### Interactive CLI Mode
```bash
./ptx_vm
> load examples/control_flow_example.ptx
> run
> dump
> exit
```

## Command Reference

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

### visualize
Display visualization of execution state.
```bash
> visualize <type>
```

Where `<type>` can be:
- `warp` - Warp execution visualization
- `memory` - Memory access visualization
- `performance` - Performance counter display

### quit
Exit the virtual machine.
```bash
> quit
```

## Documentation
Comprehensive documentation is available in the `docs/` directory:
- [User Guide](docs/user_guide.md) - For end-users and application developers
- [Developer Guide](docs/developer_guide.md) - For contributors and developers
- [API Documentation](docs/api_documentation.md) - For developers integrating with the VM
- [Reconvergence Mechanism](docs/reconvergence_mechanism.md) - Technical details on divergence handling
- [Memory Optimizations](docs/memory_optimizations.md) - Details on memory system implementation
- [CUDA Binary Loader](docs/cuda_binary_loader.md) - Technical details on CUDA binary loading
- [Performance Testing](docs/performance_testing.md) - Performance testing framework and results
- [Warp Scheduler](docs/warp_scheduler.md) - Technical details on warp scheduling implementation
- [Predicate Handler](docs/predicate_handler.md) - Technical details on predicate handling implementation
- [Divergence Handling](docs/divergence_handling.md) - Technical details on divergence handling and reconvergence algorithms
- [Memory Optimization Examples](docs/memory_optimization_examples.md) - Example programs for memory optimization testing
- [Divergence Performance Testing](docs/divergence_performance_testing.md) - Performance testing framework for divergence handling
- [Visualization Features](docs/visualization_features.md) - Details on visualization capabilities

The documentation covers architecture, code structure, contribution guidelines, and technical details of implementation.

## Release Information

### Release Notes
See [RELEASE_NOTES.md](RELEASE_NOTES.md) for information about this release, including:
- Key features
- Installation instructions
- Usage examples
- Known issues
- Future enhancements

### Contributors
See [CONTRIBUTORS.md](CONTRIBUTORS.md) for a list of contributors and information about how to contribute to the project.

### License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Author
- **Zhenzhong Han** - Lead Developer and Architect
  - Email: zhenzhong.han@qq.com
  - Role: Chief architect and main developer of the PTX Virtual Machine