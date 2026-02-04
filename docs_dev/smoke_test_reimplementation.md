# Smoke Test 重新实现说明

## 概述

由于功能源码更新，完全重新实现了 `tests/system_tests/smoke_test.cpp`，以匹配当前的 PTX VM 架构和 API。

## 主要变更

### 1. 使用新的加载/执行流程

**旧代码：**
```cpp
bool result = vm->loadAndExecuteProgram("examples/simple_math_example.ptx");
```

**新代码：**
```cpp
// 分离加载和执行
bool loaded = vm->loadProgram("examples/simple_math_example.ptx");
ASSERT_TRUE(loaded);

// 设置内存和参数
CUdeviceptr resultPtr = vm->allocateMemory(20);
std::vector<KernelParameter> params;
params.push_back({resultPtr, sizeof(uint64_t), 0});
vm->setKernelParameters(params);

// 执行程序
bool executed = vm->run();
EXPECT_TRUE(executed);
```

### 2. 正确的内存管理

新测试使用完整的 CUDA-like 内存管理流程：

```cpp
// 分配设备内存
CUdeviceptr ptr = vm->allocateMemory(size);

// 拷贝数据到设备
vm->copyMemoryHtoD(devicePtr, hostData, size);

// 拷贝数据回主机
vm->copyMemoryDtoH(hostData, devicePtr, size);

// 释放内存
vm->freeMemory(ptr);
```

### 3. 实际验证计算结果

不再只是检查指令是否执行，而是验证实际的计算结果：

```cpp
// 读回结果
int32_t results[5];
vm->copyMemoryDtoH(results, resultPtr, sizeof(results));

// 验证算术结果（来自 simple_math_example.ptx）
EXPECT_EQ(results[0], 49);   // 42 + 7
EXPECT_EQ(results[1], 35);   // 42 - 7
EXPECT_EQ(results[2], 294);  // 42 * 7
EXPECT_EQ(results[3], 6);    // 42 / 7
EXPECT_EQ(results[4], 0);    // 42 % 7
```

## 新增测试用例

### 1. TestBasicProgramExecution
- 测试基本的程序加载、执行和结果验证
- 验证性能计数器功能
- 验证实际的算术计算结果

### 2. TestControlFlowExecution
- 测试控制流（分支、循环）
- 验证分支统计
- 验证循环计算结果（输入值 × 5）

### 3. TestMemoryOperations
- 测试内存加载和存储操作
- 验证全局内存读写计数
- 验证数据传输的正确性

### 4. TestSystemIntegration
- 综合测试所有组件的集成
- 验证寄存器、内存、执行器协同工作
- 打印详细的性能指标（IPC 等）

### 5. TestProgramParsing
- 测试 PTX 程序解析
- 验证元数据解析（版本、目标架构）
- 验证符号表和函数结构
- 验证入口内核识别

### 6. TestRegisterOperations
- 测试寄存器的基本读写
- 测试多个寄存器的批量操作
- 验证寄存器在程序执行中的使用
- 统计寄存器读写次数

### 7. TestPerformanceCounters
- 测试性能计数器的重置功能
- 验证各种计数器的更新
- 打印完整的性能计数器摘要

### 8. TestMemoryManagement
- 测试内存分配和释放
- 测试多个内存分配的独立性
- 测试主机到设备、设备到主机的数据传输
- 验证数据完整性
- 检测内存泄漏

## 删除的测试

移除了以下不再适用的测试：
- `TestSchedulingAlgorithms` - InstructionScheduler API 已更改
- 旧的 `TestRegisterAllocation` - RegisterAllocator API 已简化

## 关键改进

### 1. 测试独立性
每个测试都有完整的设置、执行、验证和清理流程。

### 2. 详细的断言消息
```cpp
ASSERT_TRUE(loaded) << "Failed to load program";
EXPECT_GT(instructionsExecuted, 0u) << "No instructions were executed";
```

### 3. 资源清理
所有测试都正确释放分配的内存：
```cpp
// Clean up
vm->freeMemory(resultPtr);
```

### 4. TearDown 方法
添加了 TearDown 方法确保每个测试后清理 VM：
```cpp
void TearDown() override {
    vm.reset();
}
```

## 使用的 API

### PTXVM 核心 API
- `loadProgram(filename)` - 加载 PTX 程序
- `run()` - 执行程序
- `isProgramLoaded()` - 检查是否已加载程序
- `allocateMemory(size)` - 分配设备内存
- `freeMemory(ptr)` - 释放设备内存
- `copyMemoryHtoD()` - 主机到设备拷贝
- `copyMemoryDtoH()` - 设备到主机拷贝
- `setKernelParameters()` - 设置内核参数
- `getMemoryAllocations()` - 获取内存分配信息

### 组件访问器
- `getPerformanceCounters()` - 获取性能计数器
- `getRegisterBank()` - 获取寄存器组
- `getRegisterAllocator()` - 获取寄存器分配器
- `getExecutor()` - 获取执行器
- `getMemorySubsystem()` - 获取内存子系统

### 性能计数器
- `INSTRUCTIONS_EXECUTED` - 已执行指令数
- `CYCLES` - 周期数
- `REGISTER_READS/WRITES` - 寄存器读写
- `GLOBAL_MEMORY_READS/WRITES` - 全局内存读写
- `PARAMETER_MEMORY_READS` - 参数内存读取
- `BRANCHES` - 分支数

## 测试覆盖范围

新的测试套件覆盖了：
1. ✅ 程序加载和解析
2. ✅ 程序执行
3. ✅ 内存管理（分配、拷贝、释放）
4. ✅ 参数传递
5. ✅ 寄存器操作
6. ✅ 算术指令
7. ✅ 控制流指令（分支、循环）
8. ✅ 内存访问指令
9. ✅ 性能计数器
10. ✅ 符号表和程序结构
11. ✅ 数据完整性验证

## 编译和运行

```bash
# 构建测试
cd build
cmake ..
make smoke_test

# 运行测试
./tests/system_tests/smoke_test

# 或使用 CTest
ctest -R SystemSmokeTest -V
```

## 预期输出

每个测试应该输出详细的执行信息：

```
[==========] Running 8 tests from 1 test suite.
[----------] 8 tests from SystemSmokeTest
[ RUN      ] SystemSmokeTest.TestBasicProgramExecution
[       OK ] SystemSmokeTest.TestBasicProgramExecution
[ RUN      ] SystemSmokeTest.TestControlFlowExecution
[       OK ] SystemSmokeTest.TestControlFlowExecution
...

=== System Integration Test Results ===
Total instructions executed: 42
Total execution cycles: 50
Instructions per cycle (IPC): 0.84

=== Performance Counter Summary ===
INSTRUCTIONS_EXECUTED: 42
CYCLES: 50
REGISTER_READS: 120
REGISTER_WRITES: 85
...
```

## 未来改进

1. 添加更多复杂程序的测试
2. 测试多函数调用
3. 测试共享内存操作
4. 测试原子操作
5. 性能基准测试
6. 压力测试（大规模数据）

## 相关文档

- [loadAndExecuteProgram 修复](./archive/loadAndExecuteProgram_fix.md)
- [开发者指南](./developer_guide.md)
- [Smoke Test 介绍](../blog/smoke_test_introduction.md)
