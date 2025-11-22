# PTX VM 正确使用示例

**Authors**: Han-Zhenzhong, TongyiLingma, GitHub Copilot  
**Last Updated**: 2025-10-29

本文档展示如何正确使用 PTX VM 执行 PTX 代码。

## 核心原则

**PTX kernel 需要的是 device memory 地址，而不是直接的数值！**

这与 CUDA 编程模型完全一致：
1. 在 Host (CPU) 端准备数据
2. 分配 Device (GPU) 内存
3. 拷贝数据 Host → Device
4. 启动 kernel（传递 device 地址）
5. 拷回结果 Device → Host
6. 释放内存

## 示例 1：CLI 交互式使用（向量加法）

假设我们有一个 PTX kernel `vecAdd.ptx`:

```ptx
.version 7.0
.target sm_50
.address_size 64

.entry vecAdd(
    .param .u64 A_ptr,
    .param .u64 B_ptr,
    .param .u64 C_ptr,
    .param .u32 N
)
{
    .reg .u32 %tid;
    .reg .u64 %rd<10>;
    .reg .f32 %f<5>;
    .reg .u32 %r<5>;
    
    // 获取线程 ID
    mov.u32 %tid, %tid.x;
    
    // 加载参数
    ld.param.u64 %rd1, [A_ptr];
    ld.param.u64 %rd2, [B_ptr];
    ld.param.u64 %rd3, [C_ptr];
    ld.param.u32 %r1, [N];
    
    // 检查边界
    setp.ge.u32 %p1, %tid, %r1;
    @%p1 bra DONE;
    
    // 计算地址
    mul.wide.u32 %rd4, %tid, 4;  // offset = tid * sizeof(float)
    add.u64 %rd5, %rd1, %rd4;    // A_addr = A_ptr + offset
    add.u64 %rd6, %rd2, %rd4;    // B_addr = B_ptr + offset
    add.u64 %rd7, %rd3, %rd4;    // C_addr = C_ptr + offset
    
    // 加载数据
    ld.global.f32 %f1, [%rd5];
    ld.global.f32 %f2, [%rd6];
    
    // 计算
    add.f32 %f3, %f1, %f2;
    
    // 存储结果
    st.global.f32 [%rd7], %f3;
    
DONE:
    ret;
}
```

### CLI 操作步骤

```bash
$ ./ptx_vm

# 1. 加载 PTX 程序
ptx-vm> load examples/vecAdd.ptx
Program loaded successfully.

# 2. 分配第一个输入数组的内存（8 个 float，32 字节）
ptx-vm> alloc 32
Allocated 32 bytes at address 0x10000

# 3. 填充第一个输入数组的数据
ptx-vm> fill 0x10000 8 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0
Filled 8 float values starting at 0x10000

# 4. 分配第二个输入数组的内存
ptx-vm> alloc 32
Allocated 32 bytes at address 0x10020

# 5. 填充第二个输入数组的数据
ptx-vm> fill 0x10020 8 10.0 20.0 30.0 40.0 50.0 60.0 70.0 80.0
Filled 8 float values starting at 0x10020

# 6. 分配输出数组的内存
ptx-vm> alloc 32
Allocated 32 bytes at address 0x10040

# 7. 启动 kernel（传递三个 device 地址 + 数组大小）
ptx-vm> launch vecAdd 0x10000 0x10020 0x10040 8
Parameter 0: device address 0x10000
Parameter 1: device address 0x10020
Parameter 2: device address 0x10040
Parameter 3: value 8

Launching kernel: vecAdd
Grid dimensions: 1 x 1 x 1
Block dimensions: 32 x 1 x 1

✓ Kernel launched successfully
Use 'memory <address> <size>' to view results

# 8. 查看结果
ptx-vm> memory 0x10040 32
Memory Contents at 0x10040:
  [0x10040] 11.0  (1.0 + 10.0)
  [0x10044] 22.0  (2.0 + 20.0)
  [0x10048] 33.0  (3.0 + 30.0)
  [0x1004c] 44.0  (4.0 + 40.0)
  [0x10050] 55.0  (5.0 + 50.0)
  [0x10054] 66.0  (6.0 + 60.0)
  [0x10058] 77.0  (7.0 + 70.0)
  [0x1005c] 88.0  (8.0 + 80.0)

# 9. 完成
ptx-vm> quit
```

## 示例 2：使用 Host API（推荐）

创建文件 `test_vecadd.cpp`:

```cpp
#include "host_api.hpp"
#include <iostream>
#include <vector>

int main() {
    // 1. 初始化 PTX VM
    HostAPI hostAPI;
    if (!hostAPI.initialize()) {
        std::cerr << "Failed to initialize VM" << std::endl;
        return 1;
    }
    
    // 2. 加载 PTX 程序
    if (!hostAPI.loadPTXProgram("examples/vecAdd.ptx")) {
        std::cerr << "Failed to load PTX program" << std::endl;
        return 1;
    }
    
    // 3. 准备 Host 数据
    const size_t N = 8;
    std::vector<float> h_A = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    std::vector<float> h_B = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f, 70.0f, 80.0f};
    std::vector<float> h_C(N);
    
    // 4. 分配 Device 内存
    CUdeviceptr d_A, d_B, d_C;
    hostAPI.cuMemAlloc(&d_A, N * sizeof(float));
    hostAPI.cuMemAlloc(&d_B, N * sizeof(float));
    hostAPI.cuMemAlloc(&d_C, N * sizeof(float));
    
    std::cout << "Allocated device memory:" << std::endl;
    std::cout << "  d_A = 0x" << std::hex << d_A << std::dec << std::endl;
    std::cout << "  d_B = 0x" << std::hex << d_B << std::dec << std::endl;
    std::cout << "  d_C = 0x" << std::hex << d_C << std::dec << std::endl;
    
    // 5. 拷贝数据 Host → Device
    hostAPI.cuMemcpyHtoD(d_A, h_A.data(), N * sizeof(float));
    hostAPI.cuMemcpyHtoD(d_B, h_B.data(), N * sizeof(float));
    
    std::cout << "Copied input data to device" << std::endl;
    
    // 6. 准备 kernel 参数（device 地址）
    uint32_t n = N;
    void* args[] = { 
        &d_A,   // 参数 0：A 数组的 device 地址
        &d_B,   // 参数 1：B 数组的 device 地址
        &d_C,   // 参数 2：C 数组的 device 地址
        &n      // 参数 3：数组大小
    };
    
    // 7. 启动 kernel
    std::cout << "Launching kernel..." << std::endl;
    CUresult result = hostAPI.cuLaunchKernel(
        1,           // function handle
        1, 1, 1,     // grid dimensions
        N, 1, 1,     // block dimensions (每个元素一个线程)
        0,           // shared memory
        nullptr,     // stream
        args,        // kernel parameters
        nullptr      // extra
    );
    
    if (result != CUDA_SUCCESS) {
        std::cerr << "Kernel launch failed with error code: " << result << std::endl;
        return 1;
    }
    
    std::cout << "Kernel executed successfully" << std::endl;
    
    // 8. 拷贝结果 Device → Host
    hostAPI.cuMemcpyDtoH(h_C.data(), d_C, N * sizeof(float));
    
    std::cout << "Copied results from device" << std::endl;
    
    // 9. 显示结果
    std::cout << "\nResults:" << std::endl;
    for (size_t i = 0; i < N; ++i) {
        std::cout << "  C[" << i << "] = " << h_A[i] << " + " << h_B[i] 
                  << " = " << h_C[i] << std::endl;
    }
    
    // 10. 清理内存
    hostAPI.cuMemFree(d_A);
    hostAPI.cuMemFree(d_B);
    hostAPI.cuMemFree(d_C);
    
    std::cout << "\nMemory freed" << std::endl;
    
    return 0;
}
```

编译和运行：

```bash
$ g++ -o test_vecadd test_vecadd.cpp -Iinclude -Lbuild -lptx_vm
$ ./test_vecadd

Allocated device memory:
  d_A = 0x10000
  d_B = 0x10020
  d_C = 0x10040
Copied input data to device
Launching kernel...
Kernel executed successfully
Copied results from device

Results:
  C[0] = 1 + 10 = 11
  C[1] = 2 + 20 = 22
  C[2] = 3 + 30 = 33
  C[3] = 4 + 40 = 44
  C[4] = 5 + 50 = 55
  C[5] = 6 + 60 = 66
  C[6] = 7 + 70 = 77
  C[7] = 8 + 80 = 88
```

## 示例 3：从文件加载数据

```bash
# 1. 创建输入数据文件
$ python3 -c "import struct; open('input_a.bin', 'wb').write(struct.pack('8f', *range(1,9)))"
$ python3 -c "import struct; open('input_b.bin', 'wb').write(struct.pack('8f', *[x*10 for x in range(1,9)]))"

# 2. 使用 CLI
$ ./ptx_vm

ptx-vm> load examples/vecAdd.ptx

# 分配内存
ptx-vm> alloc 32
0x10000
ptx-vm> alloc 32
0x10020
ptx-vm> alloc 32
0x10040

# 从文件加载数据
ptx-vm> loadfile 0x10000 input_a.bin 32
Loaded 32 bytes from input_a.bin to address 0x10000

ptx-vm> loadfile 0x10020 input_b.bin 32
Loaded 32 bytes from input_b.bin to address 0x10020

# 启动 kernel
ptx-vm> launch vecAdd 0x10000 0x10020 0x10040 8

# 查看结果
ptx-vm> memory 0x10040 32
```

## 常见错误

### ❌ 错误 1：直接传递数值

```bash
# 这样不会工作！
$ ./ptx_vm vecAdd.ptx 1 2 3 4 5 6 7 8
```

**为什么错误**：PTX kernel 期望的是内存地址，而不是数值。

### ❌ 错误 2：混淆 host 和 device 内存

```cpp
// 错误的代码
float h_A[] = {1, 2, 3, 4};
void* args[] = { h_A };  // ❌ 传递 host 地址
```

**正确的做法**：

```cpp
// 正确的代码
float h_A[] = {1, 2, 3, 4};
CUdeviceptr d_A;
cuMemAlloc(&d_A, 4 * sizeof(float));
cuMemcpyHtoD(d_A, h_A, 4 * sizeof(float));
void* args[] = { &d_A };  // ✓ 传递 device 地址的指针
```

### ❌ 错误 3：忘记分配输出内存

```bash
ptx-vm> alloc 32      # 只分配了输入
0x10000
ptx-vm> launch kernel 0x10000 0x10020  # ❌ 0x10020 未分配
```

**正确的做法**：

```bash
ptx-vm> alloc 32      # 输入
0x10000
ptx-vm> alloc 32      # 输出（必须分配！）
0x10020
ptx-vm> launch kernel 0x10000 0x10020  # ✓
```

## PTX 参数传递机制详解

### 内存布局

```
┌────────────────────── Host Memory (CPU) ──────────────────────┐
│                                                                │
│  float h_A[] = {1.0, 2.0, 3.0, 4.0};                          │
│  float h_B[] = {10.0, 20.0, 30.0, 40.0};                      │
│  float h_C[4];                                                │
│                                                                │
└────────────────────────────────────────────────────────────────┘
                    ↓ cuMemcpyHtoD        ↑ cuMemcpyDtoH
┌────────────────────── Device Memory (GPU) ────────────────────┐
│                                                                │
│  0x10000: [1.0, 2.0, 3.0, 4.0]     ← d_A 指向这里              │
│  0x10010: [10.0, 20.0, 30.0, 40.0] ← d_B 指向这里              │
│  0x10020: [?, ?, ?, ?]              ← d_C 指向这里（输出）      │
│                                                                │
└────────────────────────────────────────────────────────────────┘
                    ↓ kernel 通过 ld.param 和 ld.global 访问
┌────────────────── Parameter Memory (0x1000) ──────────────────┐
│                                                                │
│  offset 0:  0x0000000000010000  (d_A 的值)                    │
│  offset 8:  0x0000000000010010  (d_B 的值)                    │
│  offset 16: 0x0000000000010020  (d_C 的值)                    │
│  offset 24: 0x00000004          (N 的值 = 4)                  │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### PTX 代码中的参数访问

```ptx
.entry vecAdd(
    .param .u64 A_ptr,    // offset 0 in parameter memory
    .param .u64 B_ptr,    // offset 8
    .param .u64 C_ptr,    // offset 16
    .param .u32 N         // offset 24
)
{
    .reg .u64 %rd1, %rd2, %rd3;
    .reg .f32 %f1, %f2, %f3;
    
    // 从参数内存读取指针
    ld.param.u64 %rd1, [A_ptr];  // %rd1 = 0x10000
    ld.param.u64 %rd2, [B_ptr];  // %rd2 = 0x10010
    ld.param.u64 %rd3, [C_ptr];  // %rd3 = 0x10020
    
    // 从全局内存加载实际数据
    ld.global.f32 %f1, [%rd1];   // %f1 = memory[0x10000] = 1.0
    ld.global.f32 %f2, [%rd2];   // %f2 = memory[0x10010] = 10.0
    
    // 计算
    add.f32 %f3, %f1, %f2;       // %f3 = 11.0
    
    // 存储到全局内存
    st.global.f32 [%rd3], %f3;   // memory[0x10020] = 11.0
    
    ret;
}
```

## 总结

**关键要点**：

1. ✅ PTX kernel 接收的是 **device memory 地址**
2. ✅ 必须先 **分配 device memory**
3. ✅ 必须先 **拷贝数据到 device**
4. ✅ 启动 kernel 时传递 **地址**，而不是值
5. ✅ 结果需要 **拷回 host**
6. ✅ 最后要 **释放内存**

**推荐方式**：

- 简单测试：使用 CLI 交互式命令
- 程序化使用：使用 Host API（参考 `examples/parameter_passing_example.cpp`）
- 复杂应用：使用 Host API + 配置文件

**参考资料**：

- `docs/how_CudaC_and_PTX_called_by_HostC.md` - CUDA 调用模型详解
- `docs/cli_usage_correction.md` - CLI 使用纠正说明
- `examples/parameter_passing_example.cpp` - Host API 完整示例
- `docs/api_documentation.md` - API 文档
