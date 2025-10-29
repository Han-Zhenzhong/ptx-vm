# PTX 虚拟机使用指南

**Authors**: Han-Zhenzhong, TongyiLingma, GitHub Copilot  
**Last Updated**: 2025-10-29

## 目录
- [简介](#简介)
- [快速开始](#快速开始)
- [三种使用方式](#三种使用方式)
- [API 编程接口](#api-编程接口)
- [命令行交互模式](#命令行交互模式)
- [直接执行模式](#直接执行模式)
- [完整示例](#完整示例)
- [常见问题](#常见问题)

---

## 简介

PTX 虚拟机是一个用于执行 NVIDIA PTX（Parallel Thread Execution）中间代码的虚拟机实现。它提供了完整的 SIMT 执行模型、内存管理、性能分析等功能。

### 主要特性

- ✅ **完整的 SIMT 执行模型** - 支持并行线程执行
- ✅ **Warp 调度** - 动态线程掩码管理
- ✅ **谓词执行** - 条件操作支持
- ✅ **分歧处理** - 多种重聚算法
- ✅ **内存系统** - 分层内存模型、缓存模拟
- ✅ **性能计数器** - 详细的执行统计
- ✅ **调试支持** - 断点、监视点、单步执行

---

## 快速开始

### 1. 构建项目

```bash
# 克隆仓库
git clone https://gitee.com/hanzhenzhong/ptx-vm.git
cd ptx-vm

# 创建构建目录
mkdir build && cd build

# 配置并编译
cmake ..
make

# 运行测试（可选）
make test
```

### 2. 运行第一个示例

```bash
# 直接执行 PTX 文件
./ptx_vm ../examples/simple_math_example.ptx

# 或运行示例程序
./execution_result_demo
./parameter_passing_example
```

---

## 三种使用方式

PTX 虚拟机提供三种使用方式，适用于不同的场景：

### 方式 1：API 编程接口
**适用场景**：将 PTX-VM 集成到你的应用程序中

### 方式 2：命令行交互模式（CLI）
**适用场景**：调试、实验、学习 PTX

### 方式 3：直接执行模式
**适用场景**：快速运行 PTX 文件

---

## API 编程接口

### 基本使用流程

```cpp
#include "host_api.hpp"
#include <iostream>

int main() {
    // 1. 创建 Host API 对象
    HostAPI hostAPI;
    
    // 2. 初始化虚拟机
    if (!hostAPI.initialize()) {
        std::cerr << "初始化失败" << std::endl;
        return 1;
    }
    
    // 3. 分配内存
    CUdeviceptr devicePtr;
    CUresult result = hostAPI.cuMemAlloc(&devicePtr, 1024);
    
    // 4. 拷贝数据到设备
    std::vector<int> data(256, 42);
    hostAPI.cuMemcpyHtoD(devicePtr, data.data(), data.size() * sizeof(int));
    
    // 5. 加载并执行程序
    hostAPI.loadProgram("my_kernel.ptx");
    hostAPI.run();
    
    // 6. 拷贝结果回主机
    std::vector<int> result_data(256);
    hostAPI.cuMemcpyDtoH(result_data.data(), devicePtr, 256 * sizeof(int));
    
    // 7. 释放内存
    hostAPI.cuMemFree(devicePtr);
    
    return 0;
}
```

### 核心 API 函数

#### 内存管理

```cpp
// 分配设备内存
CUresult cuMemAlloc(CUdeviceptr* dptr, size_t bytesize);

// 释放设备内存
CUresult cuMemFree(CUdeviceptr dptr);

// 主机到设备内存拷贝
CUresult cuMemcpyHtoD(CUdeviceptr dstDevice, 
                      const void* srcHost, 
                      size_t ByteCount);

// 设备到主机内存拷贝
CUresult cuMemcpyDtoH(void* dstHost, 
                      CUdeviceptr srcDevice, 
                      size_t ByteCount);

// 设备到设备内存拷贝
CUresult cuMemcpyDtoD(CUdeviceptr dstDevice, 
                      CUdeviceptr srcDevice, 
                      size_t ByteCount);
```

#### 程序加载与执行

```cpp
// 加载 PTX 程序
bool loadProgram(const std::string& filename);

// 检查程序是否已加载
bool isProgramLoaded() const;

// 执行程序
bool run();

// 单步执行
bool step();
```

#### 调试功能

```cpp
// 设置断点
bool setBreakpoint(size_t address);

// 设置监视点
bool setWatchpoint(uint64_t address);

// 打印寄存器
void printRegisters() const;
void printAllRegisters() const;
void printPredicateRegisters() const;

// 打印内存
void printMemory(uint64_t address, size_t size) const;

// 打印性能计数器
void printPerformanceCounters() const;
```

#### 内核启动

```cpp
// 设置内核名称
void setKernelName(const std::string& name);

// 设置内核启动参数
void setKernelLaunchParams(const KernelLaunchParams& params);

// 设置内核参数
void setKernelParameters(const std::vector<KernelParameter>& parameters);

// 启动内核
bool launchKernel();
```

### 完整示例：向量加法

```cpp
#include "host_api.hpp"
#include <iostream>
#include <vector>

int main() {
    // 初始化
    HostAPI hostAPI;
    if (!hostAPI.initialize()) {
        std::cerr << "初始化失败" << std::endl;
        return 1;
    }
    
    const size_t N = 1024;
    const size_t size = N * sizeof(float);
    
    // 准备主机数据
    std::vector<float> h_A(N), h_B(N), h_C(N);
    for (size_t i = 0; i < N; i++) {
        h_A[i] = static_cast<float>(i);
        h_B[i] = static_cast<float>(i * 2);
    }
    
    // 分配设备内存
    CUdeviceptr d_A, d_B, d_C;
    hostAPI.cuMemAlloc(&d_A, size);
    hostAPI.cuMemAlloc(&d_B, size);
    hostAPI.cuMemAlloc(&d_C, size);
    
    // 拷贝输入数据到设备
    hostAPI.cuMemcpyHtoD(d_A, h_A.data(), size);
    hostAPI.cuMemcpyHtoD(d_B, h_B.data(), size);
    
    // 设置内核启动参数
    KernelLaunchParams params;
    params.kernelName = "vecAdd";
    params.gridDimX = 8;    params.gridDimY = 1;    params.gridDimZ = 1;
    params.blockDimX = 128; params.blockDimY = 1;   params.blockDimZ = 1;
    params.sharedMemBytes = 0;
    
    hostAPI.setKernelName("vecAdd");
    hostAPI.setKernelLaunchParams(params);
    
    // 设置内核参数
    std::vector<KernelParameter> kernelParams;
    KernelParameter param1 = {d_A, sizeof(CUdeviceptr), 0};
    KernelParameter param2 = {d_B, sizeof(CUdeviceptr), 8};
    KernelParameter param3 = {d_C, sizeof(CUdeviceptr), 16};
    kernelParams.push_back(param1);
    kernelParams.push_back(param2);
    kernelParams.push_back(param3);
    
    hostAPI.setKernelParameters(kernelParams);
    
    // 加载并执行内核
    if (!hostAPI.loadProgram("vecAdd.ptx")) {
        std::cerr << "加载程序失败" << std::endl;
        return 1;
    }
    
    if (!hostAPI.launchKernel()) {
        std::cerr << "内核启动失败" << std::endl;
        return 1;
    }
    
    // 拷贝结果回主机
    hostAPI.cuMemcpyDtoH(h_C.data(), d_C, size);
    
    // 验证结果
    bool correct = true;
    for (size_t i = 0; i < N; i++) {
        if (std::abs(h_C[i] - (h_A[i] + h_B[i])) > 1e-5) {
            correct = false;
            break;
        }
    }
    
    std::cout << "结果 " << (correct ? "正确" : "错误") << std::endl;
    
    // 清理
    hostAPI.cuMemFree(d_A);
    hostAPI.cuMemFree(d_B);
    hostAPI.cuMemFree(d_C);
    
    return 0;
}
```

---

## 命令行交互模式

### 启动交互模式

```bash
./ptx_vm
```

你会看到提示符：
```
PTX Virtual Machine Interactive Mode
Type 'help' for available commands
> 
```

### 常用命令

#### 1. 加载和执行程序

```bash
# 加载 PTX 文件
> load examples/simple_math_example.ptx

# 运行程序
> run

# 单步执行
> step
> step 5    # 执行 5 条指令
```

#### 2. 内存操作

```bash
# 分配内存
> alloc 1024
Allocated memory at: 0x10000

# 写入单个字节
> write 0x10000 42

# 批量填充
> fill 0x10000 10 1 2 3 4 5

# 从文件加载数据
> loadfile 0x10000 data.bin 256

# 内存拷贝
> memcpy 0x20000 0x10000 256

# 查看内存内容
> memory 0x10000 16
Memory at 0x10000:
00000000: 01 02 03 04 05 00 00 00  00 00 00 00 00 00 00 00
```

#### 3. 调试功能

```bash
# 设置断点
> break 0x100
Breakpoint set at address 0x100

# 设置监视点
> watch 0x10000
Watchpoint set at memory address 0x10000

# 查看寄存器
> register all          # 所有寄存器
> register predicate    # 谓词寄存器
> register pc           # 程序计数器

# 查看指令列表
> list

# 可视化
> visualize warp        # Warp 执行状态
> visualize memory      # 内存访问模式
> visualize performance # 性能计数器
```

#### 4. 内核启动

```bash
# 启动内核
> launch my_kernel param1 param2 param3

# 设置网格和块大小（在启动前）
> config grid 8 1 1
> config block 128 1 1
> config shared 4096
> launch my_kernel
```

#### 5. 性能分析

```bash
# 开始性能分析
> profile output.csv

# 运行程序
> run

# 查看统计信息
> dump

# 停止性能分析
> profile stop
```

#### 6. 帮助和退出

```bash
# 获取帮助
> help
> help memory    # 特定命令的帮助

# 退出
> quit
> exit
```

### 交互式调试会话示例

```bash
# 1. 启动虚拟机
$ ./ptx_vm

# 2. 加载程序
> load examples/control_flow_example.ptx
Program loaded: control_flow_example.ptx

# 3. 分配和初始化内存
> alloc 1024
Allocated memory at: 0x10000
> fill 0x10000 10 1 2 3 4 5 6 7 8 9 10
Filled 10 bytes starting at 0x10000

# 4. 设置断点
> break 0x20
Breakpoint set at address 0x20

# 5. 开始执行
> run
Execution paused at breakpoint 0x20

# 6. 检查状态
> register all
R0: 0x0000000000000001
R1: 0x0000000000000002
...

> memory 0x10000 16
Memory at 0x10000:
00000000: 01 02 03 04 05 06 07 08  09 0a 00 00 00 00 00 00

# 7. 单步执行
> step
Executed 1 instruction at PC: 0x24

> register pc
PC: 0x28

# 8. 继续执行
> run
Program completed successfully

# 9. 查看性能统计
> dump
=== Execution Statistics ===
Instructions Executed: 1024
Cycles: 2048
Branch Instructions: 32
Memory Operations: 256
...

# 10. 可视化
> visualize performance
=== Performance Counters ===
[Performance visualization output]

# 11. 退出
> quit
```

---

## 直接执行模式

### 基本用法

最简单的使用方式是直接执行 PTX 文件：

```bash
./ptx_vm examples/simple_math_example.ptx
```

### 带参数执行

```bash
# 执行并显示详细输出
./ptx_vm -v examples/control_flow_example.ptx

# 执行并保存性能报告
./ptx_vm -p report.csv examples/memory_ops_example.ptx

# 设置调试级别
./ptx_vm -d 2 examples/simple_math_example.ptx
```

### 命令行选项

```
用法: ptx_vm [选项] <ptx_file>

选项:
  -h, --help              显示帮助信息
  -v, --verbose           详细输出模式
  -d, --debug <level>     设置调试级别 (0-3)
  -p, --profile <file>    启用性能分析并保存到文件
  -i, --interactive       加载后进入交互模式
  -s, --step              单步执行模式
  --no-optimize          禁用优化
```

### 示例

```bash
# 简单执行
./ptx_vm examples/simple_math_example.ptx

# 详细输出
./ptx_vm -v examples/simple_math_example.ptx

# 性能分析
./ptx_vm -p performance.csv examples/memory_ops_example.ptx

# 加载后进入交互模式
./ptx_vm -i examples/control_flow_example.ptx

# 单步执行模式
./ptx_vm -s examples/simple_math_example.ptx
```

---

## 完整示例

### 示例 1：参数传递

```cpp
// parameter_passing_example.cpp
#include "host_api.hpp"
#include <iostream>
#include <vector>

int main() {
    HostAPI hostAPI;
    hostAPI.initialize();
    
    // 分配内存
    const size_t dataSize = 1024;
    CUdeviceptr inputPtr, outputPtr;
    
    hostAPI.cuMemAlloc(&inputPtr, dataSize * sizeof(int));
    hostAPI.cuMemAlloc(&outputPtr, dataSize * sizeof(int));
    
    // 准备数据
    std::vector<int> inputData(dataSize);
    for (size_t i = 0; i < dataSize; ++i) {
        inputData[i] = static_cast<int>(i);
    }
    
    // 拷贝到设备
    hostAPI.cuMemcpyHtoD(inputPtr, inputData.data(), 
                         dataSize * sizeof(int));
    
    // 准备内核参数
    std::vector<KernelParameter> params;
    params.push_back({inputPtr, sizeof(CUdeviceptr), 0});
    params.push_back({outputPtr, sizeof(CUdeviceptr), 8});
    params.push_back({dataSize, sizeof(size_t), 16});
    
    // 设置启动配置
    KernelLaunchParams launchParams;
    launchParams.kernelName = "process_data";
    launchParams.gridDimX = 8;
    launchParams.blockDimX = 128;
    
    hostAPI.setKernelLaunchParams(launchParams);
    hostAPI.setKernelParameters(params);
    
    // 加载并执行
    hostAPI.loadProgram("kernel.ptx");
    hostAPI.launchKernel();
    
    // 获取结果
    std::vector<int> outputData(dataSize);
    hostAPI.cuMemcpyDtoH(outputData.data(), outputPtr, 
                         dataSize * sizeof(int));
    
    // 验证
    std::cout << "前 5 个结果: ";
    for (int i = 0; i < 5; ++i) {
        std::cout << outputData[i] << " ";
    }
    std::cout << std::endl;
    
    // 清理
    hostAPI.cuMemFree(inputPtr);
    hostAPI.cuMemFree(outputPtr);
    
    return 0;
}
```

### 示例 2：使用 PTXVM 类直接操作

```cpp
#include "vm.hpp"
#include <iostream>

int main() {
    // 创建虚拟机
    PTXVM vm;
    
    // 初始化
    if (!vm.initialize()) {
        std::cerr << "初始化失败" << std::endl;
        return 1;
    }
    
    // 加载程序
    if (!vm.loadAndExecuteProgram("examples/simple_math_example.ptx")) {
        std::cerr << "加载程序失败" << std::endl;
        return 1;
    }
    
    // 获取性能计数器
    PerformanceCounters& counters = vm.getPerformanceCounters();
    
    std::cout << "执行的指令数: " 
              << counters.getCounterValue(
                     PerformanceCounterIDs::INSTRUCTIONS_EXECUTED) 
              << std::endl;
    
    std::cout << "周期数: " 
              << counters.getCounterValue(
                     PerformanceCounterIDs::CYCLES) 
              << std::endl;
    
    // 可视化
    vm.visualizePerformance();
    
    return 0;
}
```

### 示例 3：调试器使用

```cpp
#include "vm.hpp"
#include "debugger.hpp"
#include <iostream>

int main() {
    PTXVM vm;
    vm.initialize();
    vm.loadProgram("program.ptx");
    
    // 获取调试器
    Debugger& debugger = vm.getDebugger();
    
    // 设置断点
    debugger.setBreakpoint(0x100);
    debugger.setBreakpoint(0x200);
    
    // 开始执行（在断点处暂停）
    debugger.startExecution();
    
    // 检查当前位置
    size_t currentPC = debugger.getCurrentInstructionIndex();
    std::cout << "在断点处暂停，PC = " << currentPC << std::endl;
    
    // 查看寄存器
    debugger.printRegisters();
    
    // 查看内存
    debugger.printMemory(MemorySpace::GLOBAL, 0x10000, 16);
    
    // 反汇编当前指令
    debugger.disassembleCurrent();
    
    // 单步执行
    debugger.stepInstruction();
    
    // 继续执行到下一个断点
    debugger.continueExecution();
    
    return 0;
}
```

---

## 常见问题

### Q1: 如何编译我的代码以使用 PTX-VM？

**A:** 使用 CMake 或直接链接：

```bash
# 使用 CMake
cmake_minimum_required(VERSION 3.14)
project(MyPTXApp)

# 添加 PTX-VM
add_subdirectory(ptx-vm)

add_executable(my_app main.cpp)
target_link_libraries(my_app ptx_vm)
target_include_directories(my_app PRIVATE ptx-vm/include)
```

```bash
# 直接编译
g++ -std=c++20 my_app.cpp -I ptx-vm/include -L ptx-vm/build -lptx_vm -o my_app
```

### Q2: 支持哪些 PTX 版本？

**A:** 当前支持 PTX ISA 3.0 及以上的大部分指令。具体支持的指令集请参考 `docs/instruction_types.hpp`。

### Q3: 如何生成 PTX 文件？

**A:** 从 CUDA 源代码生成 PTX：

```bash
# 使用 nvcc 编译器
nvcc -ptx kernel.cu -o kernel.ptx

# 指定计算能力
nvcc -arch=sm_75 -ptx kernel.cu -o kernel.ptx

# 查看 PTX 内容
cat kernel.ptx
```

### Q4: 如何调试执行错误？

**A:** 使用调试功能：

1. **交互模式**：
```bash
./ptx_vm -i program.ptx
> break 0x100
> run
> register all
> memory 0x10000 16
```

2. **使用调试器 API**：
```cpp
Debugger& debugger = vm.getDebugger();
debugger.setBreakpoint(suspectedAddress);
debugger.startExecution();
debugger.printRegisters();
```

3. **启用详细输出**：
```bash
./ptx_vm -v -d 3 program.ptx
```

### Q5: 性能如何？可以用于生产环境吗？

**A:** PTX-VM 主要用于：
- 教学和学习 PTX 执行模型
- CUDA 程序的功能验证
- 性能分析和优化研究
- 没有 GPU 时的程序开发

不建议在生产环境中替代真实 GPU。

### Q6: 如何查看支持的命令？

**A:** 
```bash
# 交互模式
./ptx_vm
> help
> help memory  # 特定命令帮助

# 命令行帮助
./ptx_vm --help
```

### Q7: 内存地址空间是如何组织的？

**A:** 
```
0x00000000 - 0x00000FFF  : 系统保留
0x00001000 - 0x00001FFF  : 参数内存
0x00002000 - 0x0000FFFF  : 系统保留
0x00010000 - 0xFFFFFFFF  : 可分配的全局内存
```

### Q8: 如何获取性能统计？

**A:** 
```cpp
PerformanceCounters& counters = vm.getPerformanceCounters();
size_t instructions = counters.getCounterValue(
    PerformanceCounterIDs::INSTRUCTIONS_EXECUTED);
size_t cycles = counters.getCounterValue(
    PerformanceCounterIDs::CYCLES);
```

或使用命令行：
```bash
> dump
> visualize performance
```

---

## 高级主题

### 性能分析和优化

详见文档：
- `docs/performance_testing.md` - 性能测试框架
- `docs/memory_optimizations.md` - 内存优化技术
- `docs/divergence_performance_testing.md` - 分歧处理性能

### 内部实现

详见文档：
- `docs/developer_guide.md` - 开发者指南
- `docs/warp_scheduler.md` - Warp 调度器实现
- `docs/reconvergence_mechanism.md` - 重聚机制
- `docs/predicate_handler.md` - 谓词处理器

### 扩展和定制

如需扩展虚拟机功能，参考：
- `docs/api_documentation.md` - API 文档
- `CONTRIBUTORS.md` - 贡献指南
- `HEADER_ORGANIZATION.md` - 代码组织结构

---

## 获取帮助

- **文档**：查看 `docs/` 目录下的详细文档
- **示例**：参考 `examples/` 目录下的示例代码
- **测试**：查看 `tests/` 目录了解使用方式
- **问题**：在 Gitee 上提交 Issue：https://gitee.com/hanzhenzhong/ptx-vm/issues

---

## 作者

**韩振中 (Zhenzhong Han)**
- Email: zhenzhong.han@qq.com
- Gitee: https://gitee.com/hanzhenzhong

## 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。
