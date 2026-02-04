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
- 🚀 **[用户文档](./docs_user/)** - 使用指南、API 文档、示例代码
- 🔧 **[开发文档](./docs_dev/)** - 开发者指南、实现总结、性能优化
- 📖 **[规范文档](./docs_spec/)** - PTX 基础知识、SIMT 执行模型、技术规范

主要文档：
- [用户指南](./docs_user/user_guide.md) - 快速开始使用 PTX VM
- [开发者指南](./docs_dev/developer_guide.md) - 参与项目开发
- [快速参考](./docs_user/quick_reference.md) - 常用命令速查
- [API 文档](./docs_user/api_documentation.md) - 完整 API 参考

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

The PTX VM provides **three ways to use** the virtual machine:

### 1. 🚀 Quick Start - Direct Execution Mode
Run PTX programs directly from the command line:

```bash
# Basic execution (with default INFO log level)
./ptx_vm examples/simple_math_example.ptx

# With debug logging to see detailed execution
./ptx_vm --log-level debug examples/control_flow_example.ptx

# Run example programs
cd build
./execution_result_demo
./parameter_passing_example
```

**Command-line options:**
- `-h, --help` - Display help message
- `-l, --log-level LEVEL` - Set log level: `debug`, `info` (default), `warning`, `error`

### 2. 💻 Interactive CLI Mode
For debugging, learning, and experimentation:

```bash
# Start interactive mode
./ptx_vm

# Interactive commands
> load examples/control_flow_example.ptx  # Load PTX program
> alloc 1024                              # Allocate memory
> launch myKernel 0x10000                 # Launch kernel with parameters
> memory 0x10000 256                      # View memory contents
> dump                                    # Show execution statistics
> loglevel debug                          # Change log level
> exit                                    # Exit the VM
```

### 3. 🔧 API Programming Mode
Integrate PTX VM into your application:

```cpp
#include "host_api.hpp"

int main() {
    // Initialize VM
    HostAPI hostAPI;
    hostAPI.initialize();
    
    // Allocate device memory
    CUdeviceptr devicePtr;
    hostAPI.cuMemAlloc(&devicePtr, 1024 * sizeof(int));
    
    // Prepare and copy data
    std::vector<int> data(1024, 42);
    hostAPI.cuMemcpyHtoD(devicePtr, data.data(), 1024 * sizeof(int));
    
    // Load and execute PTX program
    hostAPI.loadProgram("my_kernel.ptx");
    
    // Launch kernel with parameters
    void* params[] = { &devicePtr, &size };
    hostAPI.cuLaunchKernel(kernel, 1,1,1, 32,1,1, 0, 0, params, nullptr);
    
    // Copy results back
    std::vector<int> results(1024);
    hostAPI.cuMemcpyDtoH(results.data(), devicePtr, 1024 * sizeof(int));
    
    // Cleanup
    hostAPI.cuMemFree(devicePtr);
    return 0;
}
```

### Log Level Control

Control the verbosity of VM output:

```bash
# Command-line mode
./ptx_vm --log-level debug program.ptx    # Detailed debug info
./ptx_vm --log-level info program.ptx     # General info (default)
./ptx_vm --log-level warning program.ptx  # Warnings and errors
./ptx_vm --log-level error program.ptx    # Errors only

# Interactive mode
> loglevel debug     # Enable all logs
> loglevel info      # Default level
> loglevel warning   # Warnings and errors only
> loglevel error     # Errors only
> loglevel           # Display current level
```

**Log levels:**
- **debug** - Shows detailed execution info, register values, memory operations
- **info** - Shows program loading, kernel launches, general info (default)
- **warning** - Shows warnings and errors only
- **error** - Shows errors only

For more details, see:
- 📖 [Complete User Guide](./docs_user/user_guide.md) - Detailed usage instructions
- 📖 [中文用户指南](./docs_user/USER_GUIDE_CN.md) - Chinese user guide
- 📖 [Quick Reference](./docs_user/quick_reference.md) - Command quick reference
- 📖 [API Documentation](./docs_user/api_documentation.md) - API reference
- 📖 [Logging System](./docs_user/logging_system.md) - Logging system details

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
Comprehensive documentation is available via [DOCS_INDEX.md](DOCS_INDEX.md) and organized into three directories:

- [docs_user/](docs_user/) - End-user and API usage documentation
- [docs_dev/](docs_dev/) - Contributor/developer documentation and technical reports
- [docs_spec/](docs_spec/) - PTX/SIMT fundamentals and specification notes

Recommended starting points:
- [User Guide](docs_user/user_guide.md)
- [Developer Guide](docs_dev/developer_guide.md)
- [API Documentation](docs_user/api_documentation.md)
- [Next Phase Development Plan](docs_dev/next_phase_development_plan.md)

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