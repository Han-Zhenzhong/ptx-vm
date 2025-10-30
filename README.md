# NVIDIA PTX Virtual Machine

A virtual machine implementation for executing NVIDIA PTX (Parallel Thread Execution) intermediate code. This project provides a complete VM architecture with advanced execution features, memory optimizations, and comprehensive tooling.

**Authors**: Han-Zhenzhong, TongyiLingma, GitHub Copilot

The project is hosted on gitee at: https://gitee.com/hanzhenzhong/ptx-vm

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
- 📚 **[完整文档索引](./DOCS_INDEX.md)** - 查看所有文档分类
- 🚀 **[用户文档](./user_docs/)** - 使用指南、API 文档、示例代码
- 🔧 **[开发文档](./dev_docs/)** - 开发者指南、实现总结、性能优化
- 📖 **[规范文档](./spec_docs/)** - PTX 基础知识、SIMT 执行模型、技术规范

主要文档：
- [用户指南](./user_docs/user_guide.md) - 快速开始使用 PTX VM
- [开发者指南](./dev_docs/developer_guide.md) - 参与项目开发
- [快速参考](./user_docs/quick_reference.md) - 常用命令速查
- [API 文档](./user_docs/api_documentation.md) - 完整 API 参考

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

### Log Level Control
```bash
# Set log level via command line
./ptx_vm --log-level debug program.ptx
./ptx_vm -l info program.ptx

# Change log level in interactive mode
> loglevel debug    # Enable all logs
> loglevel info     # Default level
> loglevel warning  # Warnings and errors only
> loglevel error    # Errors only
```

Available log levels: `debug`, `info` (default), `warning`, `error`

For more details, see [Logging System Documentation](./user_docs/logging_system.md).

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

### fill
Fill memory with multiple byte values starting at a specific address.
```bash
> fill <address> <count> <value1> [value2] ...
```

### loadfile
Load data from a file into VM memory at a specific address.
```bash
> loadfile <address> <file> <size>
```

### launch
Launch a kernel with parameters.
```bash
> launch <kernel_name> [param1] [param2] ...
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
- [Next Phase Development Plan](docs/next_phase_development_plan.md) - Plans for future development

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