# PTX VM CLI 使用方式纠正说明

**Authors**: Han-Zhenzhong, TongyiLingma, GitHub Copilot  
**Last Updated**: 2025-10-29

## 问题分析

当前 CLI 的 `processArguments` 函数存在设计问题：

```cpp
void processArguments(int argc, char* argv[]) {
    if (argc > 1) {
        loadProgram(argv[1]);  // 只是加载 PTX 文件
        
        if (argc > 2) {
            // 错误：直接把命令行参数当作 kernel 参数存储
            for (int i = 2; i < argc; ++i) {
                m_kernelParams.push_back(argv[i]);
            }
        }
    }
}
```

### ❌ 错误的理解

**直接通过命令行传递数据给 PTX kernel**：
```bash
# 错误示例 - 这样不符合 CUDA 执行模型
./ptx_vm example.ptx 10 20 30
```

这种方式无法工作，因为：
1. PTX kernel 需要的是 **device memory 地址**，而不是具体的数值
2. 必须先在 VM 中分配 device memory
3. 必须将 host 数据复制到 device memory
4. 然后才能将 device memory 的地址作为参数传递给 kernel

## ✅ 正确的 CUDA/PTX 执行流程

根据 `how_CudaC_and_PTX_called_by_HostC.md` 的描述，正确流程是：

### 1. Host 端准备数据（CPU）

```cpp
// Host 端的数据
float h_A[8] = {0, 1, 2, 3, 4, 5, 6, 7};
float h_B[8] = {0, 10, 20, 30, 40, 50, 60, 70};
float h_C[8];  // 存储结果
```

### 2. 分配 Device 内存（GPU）

```cpp
// 在 VM 的全局内存中分配空间
CUdeviceptr d_A, d_B, d_C;
cuMemAlloc(&d_A, 8 * sizeof(float));  // 例如：返回地址 0x10000
cuMemAlloc(&d_B, 8 * sizeof(float));  // 例如：返回地址 0x10020
cuMemAlloc(&d_C, 8 * sizeof(float));  // 例如：返回地址 0x10040
```

### 3. 拷贝数据 Host → Device

```cpp
// 将 host 数据复制到 device 内存
cuMemcpyHtoD(d_A, h_A, 8 * sizeof(float));
cuMemcpyHtoD(d_B, h_B, 8 * sizeof(float));
```

### 4. 启动 Kernel

```cpp
// kernel 参数是 device 内存的地址
void* args[] = { 
    &d_A,  // 0x10000 的地址
    &d_B,  // 0x10020 的地址
    &d_C,  // 0x10040 的地址
    &(int){8}  // 数组大小
};

cuLaunchKernel(kernel, 1, 1, 1, 8, 1, 1, 0, 0, args, 0);
```

### 5. 拷贝结果 Device → Host

```cpp
cuMemcpyDtoH(h_C, d_C, 8 * sizeof(float));
```

### 6. 清理内存

```cpp
cuMemFree(d_A);
cuMemFree(d_B);
cuMemFree(d_C);
```

## 正确的 CLI 使用方式

### 方式 1：交互式命令（推荐）

```bash
$ ./ptx_vm

ptx-vm> load examples/simple_math_example.ptx
Program loaded successfully.

# 1. 分配内存
ptx-vm> alloc 32          # 分配 32 字节用于输入 A
Allocated 32 bytes at address 0x10000

ptx-vm> alloc 32          # 分配 32 字节用于输入 B
Allocated 32 bytes at address 0x10020

ptx-vm> alloc 32          # 分配 32 字节用于输出 C
Allocated 32 bytes at address 0x10040

# 2. 填充输入数据
ptx-vm> fill 0x10000 8 0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0
Filled 8 values starting at 0x10000

ptx-vm> fill 0x10020 8 0.0 10.0 20.0 30.0 40.0 50.0 60.0 70.0
Filled 8 values starting at 0x10020

# 3. 启动 kernel（传递 device memory 地址）
ptx-vm> launch vecAdd 0x10000 0x10020 0x10040
Launching kernel: vecAdd
Kernel launched successfully

# 4. 查看结果
ptx-vm> memory 0x10040 32
Memory Contents:
0x10040: 0.0 11.0 22.0 33.0 44.0 55.0 66.0 77.0
```

### 方式 2：从文件加载数据

```bash
ptx-vm> load examples/simple_math_example.ptx
ptx-vm> alloc 1024
Allocated 1024 bytes at address 0x10000

# 从文件加载数据到指定地址
ptx-vm> loadfile 0x10000 input_data.bin 1024
Loaded 1024 bytes from input_data.bin to address 0x10000

ptx-vm> launch myKernel 0x10000 0x11000
```

### 方式 3：使用 Host API（程序化方式）

这是最推荐的方式，参考 `examples/parameter_passing_example.cpp`：

```cpp
#include "host_api.hpp"

int main() {
    HostAPI hostAPI;
    hostAPI.initialize();
    
    // 1. 分配内存
    CUdeviceptr d_A, d_B, d_C;
    hostAPI.cuMemAlloc(&d_A, 8 * sizeof(float));
    hostAPI.cuMemAlloc(&d_B, 8 * sizeof(float));
    hostAPI.cuMemAlloc(&d_C, 8 * sizeof(float));
    
    // 2. 准备数据
    float h_A[8] = {0, 1, 2, 3, 4, 5, 6, 7};
    float h_B[8] = {0, 10, 20, 30, 40, 50, 60, 70};
    
    // 3. 拷贝到 device
    hostAPI.cuMemcpyHtoD(d_A, h_A, 8 * sizeof(float));
    hostAPI.cuMemcpyHtoD(d_B, h_B, 8 * sizeof(float));
    
    // 4. 加载 PTX
    hostAPI.loadPTXProgram("vecAdd.ptx");
    
    // 5. 准备 kernel 参数（device 地址）
    void* args[] = { &d_A, &d_B, &d_C, &(int){8} };
    
    // 6. 启动 kernel
    hostAPI.cuLaunchKernel(
        1,           // function handle
        1, 1, 1,     // grid dimensions
        8, 1, 1,     // block dimensions
        0, nullptr,  // shared memory & stream
        args, nullptr
    );
    
    // 7. 拷回结果
    float h_C[8];
    hostAPI.cuMemcpyDtoH(h_C, d_C, 8 * sizeof(float));
    
    // 8. 打印结果
    for (int i = 0; i < 8; i++) {
        printf("%f + %f = %f\n", h_A[i], h_B[i], h_C[i]);
    }
    
    // 9. 清理
    hostAPI.cuMemFree(d_A);
    hostAPI.cuMemFree(d_B);
    hostAPI.cuMemFree(d_C);
    
    return 0;
}
```

## 为什么必须这样做？

### PTX Kernel 的本质

PTX kernel 函数接收的是**指针**（内存地址），而不是具体的值：

```ptx
.entry vecAdd(
    .param .u64 A,      // 这是一个地址（指针）
    .param .u64 B,      // 这是一个地址（指针）
    .param .u64 C,      // 这是一个地址（指针）
    .param .u32 N       // 这是一个值
)
{
    // 从地址加载数据
    ld.param.u64 %rd1, [A];     // 读取 A 的值（一个地址）
    ld.global.f32 %f1, [%rd1];  // 从该地址加载实际数据
    
    // 类似地处理 B
    ld.param.u64 %rd2, [B];
    ld.global.f32 %f2, [%rd2];
    
    // 计算
    add.f32 %f3, %f1, %f2;
    
    // 存储到 C 指向的地址
    ld.param.u64 %rd3, [C];
    st.global.f32 [%rd3], %f3;
    
    ret;
}
```

### 内存层次

```
┌─────────────────────────────────────────┐
│          Host (CPU) Memory              │
│  h_A[], h_B[], h_C[]                   │
└─────────────────────────────────────────┘
           ↑ cuMemcpyHtoD ↓ cuMemcpyDtoH
┌─────────────────────────────────────────┐
│         Device (GPU) Memory             │
│                                         │
│  0x10000: [A 数据]  ← d_A 指向这里      │
│  0x10020: [B 数据]  ← d_B 指向这里      │
│  0x10040: [C 数据]  ← d_C 指向这里      │
│                                         │
└─────────────────────────────────────────┘
           ↑ kernel 通过地址访问
┌─────────────────────────────────────────┐
│        Parameter Memory (0x1000)        │
│                                         │
│  offset 0: &d_A (8 bytes) = 0x10000    │
│  offset 8: &d_B (8 bytes) = 0x10020    │
│  offset 16: &d_C (8 bytes) = 0x10040   │
│  offset 24: N (4 bytes) = 8            │
│                                         │
└─────────────────────────────────────────┘
           ↑ ld.param 从这里读取
```

## 需要修改的代码

### 1. 删除错误的命令行参数处理

`cli_interface.cpp` 的 `processArguments` 应该修改为：

```cpp
void processArguments(int argc, char* argv[]) {
    if (argc > 1) {
        // 只加载 PTX 文件
        loadProgram(argv[1]);
        
        // 如果有更多参数，显示警告
        if (argc > 2) {
            printMessage("Warning: Additional command-line arguments are ignored.");
            printMessage("PTX kernels require device memory addresses, not direct values.");
            printMessage("Use interactive commands to allocate memory and launch kernels:");
            printMessage("  1. alloc <size>        - Allocate device memory");
            printMessage("  2. fill <addr> <values> - Fill memory with data");
            printMessage("  3. launch <kernel> <addr1> <addr2> ... - Launch with addresses");
            printMessage("");
            printMessage("Or use the Host API for programmatic access.");
        }
    } else {
        printMessage("No program specified. Use 'load <filename>' to load a PTX program.");
    }
}
```

### 2. 完善 `launch` 命令

确保 `launch` 命令正确处理 device 地址：

```cpp
void launchCommand(const std::vector<std::string>& args) {
    if (args.empty()) {
        printError("Usage: launch <kernel_name> <addr1> <addr2> ...");
        printError("Example: launch vecAdd 0x10000 0x10020 0x10040");
        printError("");
        printError("Note: Arguments must be device memory addresses allocated with 'alloc'");
        return;
    }
    
    std::string kernelName = args[0];
    
    // 解析 device 地址
    std::vector<CUdeviceptr> deviceAddresses;
    for (size_t i = 1; i < args.size(); ++i) {
        try {
            CUdeviceptr addr = std::stoull(args[i], nullptr, 0);
            deviceAddresses.push_back(addr);
        } catch (...) {
            printError("Invalid address: " + args[i]);
            return;
        }
    }
    
    // 准备 kernel 参数（指向这些地址的指针）
    std::vector<void*> kernelParams;
    for (auto& addr : deviceAddresses) {
        kernelParams.push_back(&addr);
    }
    kernelParams.push_back(nullptr);
    
    // 启动 kernel
    HostAPI hostAPI;
    CUresult result = hostAPI.cuLaunchKernel(
        1, 1, 1, 1, 32, 1, 1, 0, nullptr,
        kernelParams.data(), nullptr
    );
    
    if (result == CUDA_SUCCESS) {
        printMessage("Kernel launched successfully");
    } else {
        printError("Kernel launch failed");
    }
}
```

## 完整示例对比

### ❌ 错误方式（不符合 CUDA 模型）

```bash
# 这样不会工作
./ptx_vm vecAdd.ptx 1 2 3 4 5 6 7 8
```

### ✅ 正确方式 1（CLI 交互）

```bash
./ptx_vm
> load vecAdd.ptx
> alloc 32
0x10000
> fill 0x10000 8 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0
> alloc 32
0x10020
> fill 0x10020 8 10.0 20.0 30.0 40.0 50.0 60.0 70.0 80.0
> alloc 32
0x10040
> launch vecAdd 0x10000 0x10020 0x10040
> memory 0x10040 32
```

### ✅ 正确方式 2（Host API 程序）

```cpp
// 见上面的完整 Host API 示例
```

## 总结

1. **PTX kernel 不接受直接的数值参数**，只接受 device memory 地址
2. **必须的步骤**：分配 → 拷贝 → 启动 → 读取 → 释放
3. **CLI 应该提供**：内存管理命令（alloc, fill, memcpy）而不是直接传值
4. **推荐使用**：Host API 方式，更符合 CUDA 编程模型

这就是为什么所有的 CUDA 程序都遵循 "分配-拷贝-计算-拷回-释放" 的模式！
