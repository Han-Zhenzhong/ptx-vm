# Parameter Passing Example 更新说明

## 概述

更新了 `examples/parameter_passing_example.cpp` 以匹配当前 PTX VM 的实现，正确演示参数传递机制。

## 主要变更

### 1. 程序加载

**旧代码：**
```cpp
// 未加载 PTX 程序，直接尝试启动内核
```

**新代码：**
```cpp
// 步骤 1: 加载 PTX 程序
if (!hostAPI.loadProgram("examples/parameter_passing_example.ptx")) {
    std::cerr << "Failed to load PTX program" << std::endl;
    return 1;
}
```

### 2. 参数传递方式

**旧代码：**
```cpp
// 错误：直接传递指针值
std::vector<void*> kernelParams;
kernelParams.push_back(reinterpret_cast<void*>(inputPtr));
kernelParams.push_back(reinterpret_cast<void*>(outputPtr));
kernelParams.push_back(nullptr); // Null terminator
```

**新代码：**
```cpp
// 正确：传递指向参数值的指针
std::vector<void*> kernelParams;
kernelParams.push_back(&inputPtr);      // 指向地址的指针
kernelParams.push_back(&outputPtr);     // 指向地址的指针
kernelParams.push_back(&dataSizeParam); // 指向标量值的指针
```

**关键区别：**
- 旧代码将指针值强制转换为 `void*`（错误）
- 新代码传递指向参数的指针地址（正确）
- 这与 CUDA 的 `cuLaunchKernel` API 保持一致

### 3. 参数类型对应

PTX 内核签名：
```ptx
.entry parameter_passing_kernel (
    .param .u64 input_ptr,   // 64位指针
    .param .u64 output_ptr,  // 64位指针
    .param .u32 data_size    // 32位无符号整数
)
```

C++ 参数设置：
```cpp
CUdeviceptr inputPtr;          // uint64_t
CUdeviceptr outputPtr;         // uint64_t
uint32_t dataSizeParam;        // uint32_t

// 传递指向这些变量的指针
kernelParams.push_back(&inputPtr);
kernelParams.push_back(&outputPtr);
kernelParams.push_back(&dataSizeParam);
```

### 4. 结果验证

新代码添加了详细的结果验证：

```cpp
// 验证输出数据
std::cout << "  Expected: output[i] = input[i] * 2" << std::endl;
std::cout << "    Index | Input | Output | Expected | Status" << std::endl;

for (int i = 0; i < 10; ++i) {
    int32_t expected = inputData[i] * 2;
    bool correct = (outputData[i] == expected);
    // 显示每个元素的验证结果
}
```

### 5. 错误处理

改进了错误处理和资源清理：

```cpp
if (result != CUDA_SUCCESS) {
    std::cerr << "Failed to allocate memory. Error: " << result << std::endl;
    // 清理已分配的资源
    hostAPI.cuMemFree(inputPtr);
    return 1;
}
```

### 6. 输出格式

新代码提供了结构化的步骤输出：

```
=== PTX VM Parameter Passing Example ===
========================================

Step 1: Loading PTX program...
  ✓ Program loaded successfully

Step 2: Allocating device memory...
  ✓ Allocated input memory at: 0x10000
  ✓ Allocated output memory at: 0x10020
  Data size: 256 elements

Step 3: Preparing input data...
  ✓ Created input data: [0, 1, 2, ..., 255]
  ✓ Copied 256 elements to device memory

Step 4: Preparing kernel parameters...
  Parameter 0 (input_ptr):  0x10000 (.u64 pointer)
  Parameter 1 (output_ptr): 0x10020 (.u64 pointer)
  Parameter 2 (data_size):  256 (.u32 scalar)

Step 5: Launching kernel...
  Kernel: parameter_passing_kernel
  Grid dimensions:  1 x 1 x 1
  Block dimensions: 32 x 1 x 1
  ✓ Kernel launched and executed successfully

Step 6: Retrieving results...
  ✓ Copied 256 elements from device memory

Step 7: Verifying results...
  Expected: output[i] = input[i] * 2
  First 10 elements:
    Index | Input | Output | Expected | Status
    ------|-------|--------|----------|--------
        0 |     0 |      0 |        0 | ✓
        1 |     1 |      2 |        2 | ✓
        2 |     2 |      4 |        4 | ✓
        3 |     3 |      6 |        6 | ✓
        4 |     4 |      8 |        8 | ✓
        5 |     5 |     10 |       10 | ✓
        6 |     6 |     12 |       12 | ✓
        7 |     7 |     14 |       14 | ✓
        8 |     8 |     16 |       16 | ✓
        9 |     9 |     18 |       18 | ✓

  Results: 256/256 elements correct ✓

Step 8: Cleaning up...
  ✓ Device memory freed

=== Parameter Passing Example Complete ===
SUCCESS
```

## 技术要点

### 参数内存布局

参数在参数内存空间（基地址 0x1000）中的布局：

```
地址      | 内容          | 大小  | PTX 参数
----------|--------------|------|------------------
0x1000    | inputPtr     | 8字节 | input_ptr (.u64)
0x1008    | outputPtr    | 8字节 | output_ptr (.u64)
0x1010    | dataSizeParam| 4字节 | data_size (.u32)
```

### cuLaunchKernel 实现

在 `src/host/host_api.cpp` 中：

```cpp
CUresult cuLaunchKernel(..., void** kernelParams, ...) {
    // 遍历内核参数定义
    for (size_t i = 0; i < entryFunc.parameters.size(); ++i) {
        const PTXParameter& param = entryFunc.parameters[i];
        
        // kernelParams[i] 指向参数数据
        if (kernelParams[i] != nullptr) {
            // 将参数复制到参数内存（基址 0x1000）
            const uint8_t* paramData = static_cast<const uint8_t*>(kernelParams[i]);
            for (size_t j = 0; j < param.size; ++j) {
                mem.write<uint8_t>(MemorySpace::PARAMETER, 
                                  0x1000 + offset + j, 
                                  paramData[j]);
            }
        }
        offset += param.size;
    }
}
```

### PTX 内核执行

内核通过 `ld.param` 指令加载参数：

```ptx
// 加载参数
ld.param.u64 %r1, [input_ptr];   // 从 0x1000 加载
ld.param.u64 %r2, [output_ptr];  // 从 0x1008 加载
ld.param.u32 %r3, [data_size];   // 从 0x1010 加载
```

## 与 CUDA API 的一致性

新实现与标准 CUDA Driver API 保持一致：

```cpp
// CUDA Driver API
CUdeviceptr d_ptr;
cuMemAlloc(&d_ptr, size);

void* params[] = {&d_ptr, &value};
cuLaunchKernel(func, ..., params, NULL);
```

## 编译和运行

### 编译

```bash
cd build
cmake ..
make parameter_passing_example
```

### 运行

```bash
./examples/parameter_passing_example
```

### 预期输出

程序应该：
1. ✅ 成功加载 PTX 程序
2. ✅ 分配输入和输出内存
3. ✅ 复制输入数据到设备
4. ✅ 启动内核并传递参数
5. ✅ 验证输出结果（每个输出值应该是输入值的 2 倍）
6. ✅ 清理资源

## 相关文件

- `examples/parameter_passing_example.cpp` - 主程序（已更新）
- `examples/parameter_passing_example.ptx` - PTX 内核（无需更改）
- `src/host/host_api.cpp` - Host API 实现
- `include/host_api.hpp` - Host API 头文件
- `docs_dev/archive/implementation_summary_phase1.md` - 参数传递修复文档

## 测试场景

### 场景 1：基本参数传递（256 个元素）

```
输入：[0, 1, 2, ..., 255]
输出：[0, 2, 4, ..., 510]
状态：✓ 所有元素正确
```

### 场景 2：不同数据大小

可以修改 `dataSize` 来测试不同大小：
- 小数据集：16 元素
- 中等数据集：256 元素
- 大数据集：1024 元素

### 场景 3：错误处理

测试错误情况：
- PTX 文件不存在
- 内存分配失败
- 参数数量不匹配

## 学习价值

这个示例演示了：

1. **完整的 CUDA 工作流**：
   - 初始化 → 加载程序 → 分配内存 → 复制数据 → 启动内核 → 读取结果 → 清理

2. **正确的参数传递**：
   - 指针参数（.u64）
   - 标量参数（.u32）
   - 参数内存布局

3. **错误处理和资源管理**：
   - 检查返回值
   - 清理已分配的资源
   - 提供详细的错误信息

4. **结果验证**：
   - 自动验证计算结果
   - 提供可视化输出
   - 返回适当的退出代码

## 后续改进

可能的增强：
1. 支持更多参数类型（float、double 等）
2. 支持数组参数传递
3. 添加性能计时
4. 添加命令行参数支持
5. 支持多个内核调用
