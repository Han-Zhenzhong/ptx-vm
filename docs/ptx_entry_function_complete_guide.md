# PTX 参数传递完整指南：标量 vs 指针

## 🔥 重要更正

**之前的理解（部分错误）**：
> PTX kernel 只能接收 device memory 地址作为参数，所有参数都必须先分配 device memory。

**正确的理解**：
> PTX kernel 可以接收**标量值**（整数、浮点）和**指针**（device 地址）作为参数。只有指针参数才需要先分配 device memory！

根据 `param_type_of_ptx_entry_function.md`，这是完整的说明。

---

## 一、PTX 参数的三大类型

### 1️⃣ 标量参数（Scalar Parameters）- 按值传递

**不需要分配 device memory！**

| PTX 类型 | C/C++ 类型 | 用途 | 示例 |
|----------|-----------|------|------|
| `.u8` | `uint8_t` | 小整数 | 标志位 |
| `.u16` | `uint16_t` | 短整数 | 计数器 |
| `.u32` | `uint32_t` | 整数 | 数组大小 N |
| `.s32` | `int32_t` | 有符号整数 | 偏移量 |
| `.u64` | `uint64_t` | 长整数 | 大计数 |
| `.s64` | `int64_t` | 有符号长整数 | 时间戳 |
| `.f32` | `float` | 单精度浮点 | 缩放因子 alpha |
| `.f64` | `double` | 双精度浮点 | 高精度系数 beta |

**示例**：
```ptx
.entry scalarKernel(
    .param .u32 N,        // 数组大小（值）
    .param .f32 alpha,    // 缩放因子（值）
    .param .s32 offset    // 偏移量（值）
)
{
    .reg .u32 %r1;
    .reg .f32 %f1;
    .reg .s32 %r2;
    
    ld.param.u32 %r1, [N];      // 直接读到值 1024
    ld.param.f32 %f1, [alpha];  // 直接读到值 2.5
    ld.param.s32 %r2, [offset]; // 直接读到值 -10
    
    // 使用这些值进行计算...
    ret;
}
```

**Host 调用**：
```cpp
uint32_t N = 1024;
float alpha = 2.5f;
int32_t offset = -10;

void* args[] = { &N, &alpha, &offset };
cuLaunchKernel(kernel, 1,1,1, 32,1,1, 0, 0, args, 0);
```

**关键点**：
- ✅ 直接传值，不需要 `cuMemAlloc`
- ✅ 不需要 `cuMemcpyHtoD`
- ✅ 参数在 parameter memory 中，不在 global memory

### 2️⃣ 指针参数（Pointer Parameters）- 传递地址

**需要先分配 device memory！**

```ptx
.entry pointerKernel(
    .param .u64 A_ptr,    // 指向数组 A 的指针
    .param .u64 B_ptr,    // 指向数组 B 的指针
    .param .u64 C_ptr     // 指向数组 C 的指针
)
{
    .reg .u64 %rd<5>;
    .reg .f32 %f<5>;
    
    // 读取指针（device 地址）
    ld.param.u64 %rd1, [A_ptr];  // %rd1 = 0x10000
    ld.param.u64 %rd2, [B_ptr];  // %rd2 = 0x10020
    ld.param.u64 %rd3, [C_ptr];  // %rd3 = 0x10040
    
    // 从 device memory 读取实际数据
    ld.global.f32 %f1, [%rd1];
    ld.global.f32 %f2, [%rd2];
    
    // 计算
    add.f32 %f3, %f1, %f2;
    
    // 存回 device memory
    st.global.f32 [%rd3], %f3;
    
    ret;
}
```

**Host 调用**：
```cpp
// 1. 分配 device memory
CUdeviceptr d_A, d_B, d_C;
cuMemAlloc(&d_A, N * sizeof(float));  // d_A = 0x10000
cuMemAlloc(&d_B, N * sizeof(float));  // d_B = 0x10020
cuMemAlloc(&d_C, N * sizeof(float));  // d_C = 0x10040

// 2. 拷贝数据到 device
std::vector<float> h_A(N), h_B(N);
cuMemcpyHtoD(d_A, h_A.data(), N * sizeof(float));
cuMemcpyHtoD(d_B, h_B.data(), N * sizeof(float));

// 3. 传递 device 地址
void* args[] = { &d_A, &d_B, &d_C };
cuLaunchKernel(kernel, 1,1,1, N,1,1, 0, 0, args, 0);

// 4. 拷贝结果回 host
std::vector<float> h_C(N);
cuMemcpyDtoH(h_C.data(), d_C, N * sizeof(float));

// 5. 释放 device memory
cuMemFree(d_A);
cuMemFree(d_B);
cuMemFree(d_C);
```

**关键点**：
- ✅ 必须先 `cuMemAlloc` 分配 device memory
- ✅ 必须 `cuMemcpyHtoD` 拷贝数据到 device
- ✅ 传递的是 device 地址（如 0x10000）
- ✅ PTX 中用 `ld.global` / `st.global` 访问实际数据

### 3️⃣ 结构体参数（Struct Parameters）- 按值传递

```ptx
.entry structKernel(
    .param .align 8 .b8 config[16]  // 16字节结构体
)
{
    .reg .f32 %f<5>;
    
    // 从结构体读取字段
    ld.param.f32 %f1, [config+0];   // 第1个 float
    ld.param.f32 %f2, [config+4];   // 第2个 float
    ld.param.f32 %f3, [config+8];   // 第3个 float
    ld.param.f32 %f4, [config+12];  // 第4个 float
    
    ret;
}
```

**Host 调用**：
```cpp
struct Config {
    float x, y, z, w;
};

Config cfg = {1.0f, 2.0f, 3.0f, 4.0f};
void* args[] = { &cfg };
cuLaunchKernel(kernel, 1,1,1, 1,1,1, 0, 0, args, 0);
```

---

## 二、完整示例对比

### 示例 1：纯标量参数（无需 device memory）

**场景**：计算线程的全局 ID

**PTX**：
```ptx
.entry computeGlobalId(
    .param .u32 grid_dim,     // 标量
    .param .u32 block_dim     // 标量
)
{
    .reg .u32 %r<5>;
    
    // 读取标量参数
    ld.param.u32 %r1, [grid_dim];
    ld.param.u32 %r2, [block_dim];
    
    // 读取特殊寄存器
    mov.u32 %r3, %tid.x;
    mov.u32 %r4, %ctaid.x;
    
    // 计算全局 ID = blockIdx * blockDim + threadIdx
    mad.lo.u32 %r5, %r4, %r2, %r3;
    
    // 使用 %r5 做些什么...
    
    ret;
}
```

**Host 调用**：
```cpp
uint32_t grid_dim = 10;
uint32_t block_dim = 32;

// 🌟 不需要任何 cuMemAlloc！
void* args[] = { &grid_dim, &block_dim };
cuLaunchKernel(kernel, grid_dim, 1, 1, block_dim, 1, 1, 0, 0, args, 0);

// 🌟 不需要任何 cuMemcpy！
```

### 示例 2：混合参数（指针 + 标量）

**场景**：缩放数组元素

**PTX**：
```ptx
.entry scaleArray(
    .param .u64 data_ptr,   // 指针：指向数据
    .param .u32 N,          // 标量：数组大小
    .param .f32 scale       // 标量：缩放因子
)
{
    .reg .u32 %r<5>;
    .reg .u64 %rd<5>;
    .reg .f32 %f<5>;
    .reg .pred %p1;
    
    // 读取指针参数
    ld.param.u64 %rd1, [data_ptr];  // device 地址
    
    // 读取标量参数
    ld.param.u32 %r1, [N];          // 1024
    ld.param.f32 %f1, [scale];      // 2.5
    
    // 获取线程 ID
    mov.u32 %r2, %tid.x;
    
    // 边界检查
    setp.ge.u32 %p1, %r2, %r1;
    @%p1 bra DONE;
    
    // 计算地址
    mul.wide.u32 %rd2, %r2, 4;
    add.u64 %rd3, %rd1, %rd2;
    
    // 从 device memory 读取
    ld.global.f32 %f2, [%rd3];
    
    // 缩放
    mul.f32 %f3, %f2, %f1;
    
    // 写回 device memory
    st.global.f32 [%rd3], %f3;
    
DONE:
    ret;
}
```

**Host 调用**：
```cpp
// 1. 只有数组需要 device memory
const uint32_t N = 1024;
CUdeviceptr d_data;
cuMemAlloc(&d_data, N * sizeof(float));

// 2. 拷贝数据到 device
std::vector<float> h_data(N, 1.0f);
cuMemcpyHtoD(d_data, h_data.data(), N * sizeof(float));

// 3. 准备参数
uint32_t n = N;        // 标量（值）
float scale = 2.5f;    // 标量（值）

// 4. 启动 kernel
void* args[] = {
    &d_data,  // 指针：传递 device 地址
    &n,       // 标量：传递值
    &scale    // 标量：传递值
};
cuLaunchKernel(kernel, 1,1,1, N,1,1, 0, 0, args, 0);

// 5. 读取结果
cuMemcpyDtoH(h_data.data(), d_data, N * sizeof(float));
// 结果：每个元素都是 1.0 * 2.5 = 2.5

// 6. 清理
cuMemFree(d_data);
```

### 示例 3：复杂参数组合

**PTX**：
```ptx
.entry complexKernel(
    .param .u64 input_ptr,         // 指针
    .param .u64 output_ptr,        // 指针
    .param .u32 N,                 // 标量
    .param .f32 alpha,             // 标量
    .param .align 8 .b8 cfg[8]    // 结构体
)
{
    // 读取所有参数并使用...
}
```

**Host 调用**：
```cpp
// 指针参数需要 device memory
CUdeviceptr d_in, d_out;
cuMemAlloc(&d_in, N * sizeof(float));
cuMemAlloc(&d_out, N * sizeof(float));
cuMemcpyHtoD(d_in, h_in.data(), N * sizeof(float));

// 标量和结构体直接传值
uint32_t n = N;
float alpha = 2.5f;
struct { float x, y; } cfg = {1.0f, 2.0f};

void* args[] = { &d_in, &d_out, &n, &alpha, &cfg };
cuLaunchKernel(kernel, 1,1,1, n,1,1, 0, 0, args, 0);
```

---

## 三、CLI 使用方式更正

### ❌ 之前的错误示例

```bash
# 错误：认为标量也需要分配 device memory
ptx-vm> alloc 4
0x10000
ptx-vm> fill 0x10000 1 0x04 0x00 0x00 0x00  # 存储 N=1024
ptx-vm> alloc 4
0x10004
ptx-vm> fill 0x10004 1 0x00 0x00 0x20 0x40  # 存储 alpha=2.5
ptx-vm> launch kernel 0x10000 0x10004  # ❌ 错误！
```

### ✅ 正确的方式

#### 方案 1：纯标量参数

```bash
ptx-vm> load examples/scalar_kernel.ptx
ptx-vm> launch computeId --u32 1024 --f32 2.5
#                        ↑           ↑
#                        直接传值    直接传值
✓ Kernel launched successfully
```

#### 方案 2：混合参数

```bash
# 1. 为指针参数分配 device memory
ptx-vm> alloc 4096
Allocated 4096 bytes at address 0x10000

# 2. 填充数据
ptx-vm> fill 0x10000 1024 ...
Filled 1024 float values

# 3. 启动 kernel（指针 + 标量）
ptx-vm> launch scaleArray --ptr-u64 0x10000 --u32 1024 --f32 2.5
#                         ↑                 ↑           ↑
#                         device 地址       标量值      标量值

Parameter 0: device pointer 0x10000
Parameter 1: u32 value 1024
Parameter 2: f32 value 2.5

✓ Kernel launched successfully
```

#### 方案 3：自动推断（简化版）

```bash
ptx-vm> launch kernel 0x10000 1024 2.5
#                     ↑       ↑    ↑
#                     自动识别为指针、整数、浮点
```

---

## 四、参数传递内存布局

### 完整的内存视图

```
┌────────────────── Host Memory (CPU) ─────────────────────┐
│                                                           │
│  标量值:                                                  │
│    uint32_t N = 1024;                                    │
│    float scale = 2.5f;                                   │
│                                                           │
│  数组数据:                                                │
│    std::vector<float> h_data(N);                         │
│                                                           │
│  Device 地址:                                             │
│    CUdeviceptr d_data = 0x10000;                         │
│                                                           │
└───────────────────────────────────────────────────────────┘
              ↓ cuMemcpyHtoD（只有数组需要）
┌────────────────── Device Memory (GPU) ────────────────────┐
│                                                           │
│  0x10000: [1.0, 2.0, 3.0, ...]  ← 数组数据               │
│                                                           │
└───────────────────────────────────────────────────────────┘
              ↑ ld.param获取地址，ld.global读取数据
              │
┌─────────── Parameter Memory (0x1000) ────────────────────┐
│                                                           │
│  offset 0:  0x0000000000010000  (d_data 地址)            │
│  offset 8:  0x00000400          (N = 1024 的值)          │
│  offset 12: 0x40200000          (scale = 2.5 的值)       │
│                                                           │
└───────────────────────────────────────────────────────────┘
              ↑ cuLaunchKernel 打包所有参数到这里
```

### 关键理解

1. **标量参数** → 直接在 Parameter Memory
   ```ptx
   ld.param.u32 %r1, [N];  // 直接得到 1024
   ```

2. **指针参数** → Parameter Memory 存地址，Global Memory 存数据
   ```ptx
   ld.param.u64 %rd1, [ptr];    // 得到 0x10000（地址）
   ld.global.f32 %f1, [%rd1];   // 从 0x10000 读取数据
   ```

---

## 五、改进的 CLI Launch 命令

### 推荐的命令语法

```bash
launch <kernel_name> [--type value] ...

参数类型:
  --ptr-u64 <addr>   : 64位 device 指针（需要先 alloc）
  --u32 <value>      : 32位无符号整数
  --s32 <value>      : 32位有符号整数
  --f32 <value>      : 32位浮点数
  --f64 <value>      : 64位浮点数
  --u64 <value>      : 64位无符号整数
  --s64 <value>      : 64位有符号整数
```

### 使用示例

```bash
# 示例 1：纯指针（向量加法）
ptx-vm> alloc 32
0x10000
ptx-vm> alloc 32
0x10020
ptx-vm> alloc 32
0x10040
ptx-vm> launch vecAdd --ptr-u64 0x10000 --ptr-u64 0x10020 --ptr-u64 0x10040

# 示例 2：纯标量（无需 alloc）
ptx-vm> launch computeSum --u32 100 --f32 3.14

# 示例 3：混合参数
ptx-vm> alloc 4096
0x10000
ptx-vm> launch scaleArray --ptr-u64 0x10000 --u32 1024 --f32 2.5

# 示例 4：复杂组合
ptx-vm> launch complexKernel \
        --ptr-u64 0x10000 \
        --ptr-u64 0x20000 \
        --u32 1024 \
        --f32 2.5 \
        --f64 3.14159
```

---

## 六、对比总结

### 标量 vs 指针

| 特性 | 标量参数 | 指针参数 |
|------|---------|---------|
| **PTX 类型** | `.u32`, `.f32` 等 | `.u64` |
| **传递方式** | 按值 | 按地址 |
| **Host 准备** | 直接用变量 | 需要 `cuMemAlloc` |
| **数据拷贝** | 不需要 | 需要 `cuMemcpyHtoD` |
| **PTX 读取** | `ld.param.*` 直接得到值 | `ld.param.u64` 得到地址，`ld.global` 读数据 |
| **内存位置** | Parameter Memory | Global Memory |
| **CLI 使用** | `--u32 1024` | `--ptr-u64 0x10000` |
| **何时使用** | 配置参数、大小、系数 | 大数组、批量数据 |

### 混合参数的典型模式

```cpp
// 模式 1：数组 + 大小
void* args[] = { &d_array, &N };

// 模式 2：输入 + 输出 + 参数
void* args[] = { &d_in, &d_out, &N, &alpha };

// 模式 3：多个数组 + 配置
void* args[] = { &d_A, &d_B, &d_C, &N, &scale, &offset };
```

---

## 七、关键要点

### ✅ 正确理解

1. **标量参数直接传值**，不需要分配 device memory
2. **指针参数传地址**，需要先分配 device memory
3. **Parameter Memory** 存储所有参数（值或地址）
4. **Global Memory** 只存储通过指针访问的数据
5. PTX 用 `ld.param` 读取参数，用 `ld.global` 读取数据

### ❌ 之前的错误

1. ~~所有参数都必须是 device 地址~~ ✗
2. ~~标量也需要先 alloc~~ ✗
3. ~~不能直接传递数值~~ ✗

### 🎯 实践建议

- **简单计算**：用标量参数（如求和、平均值）
- **数组处理**：指针 + 标量（指针指向数据，标量表示大小）
- **复杂算法**：混合使用，灵活组合

---

## 八、参考文档

- `docs/param_type_of_ptx_entry_function.md` - PTX 参数类型详解
- `docs/how_CudaC_and_PTX_called_by_HostC.md` - CUDA 调用模型
- `docs/cli_usage_correction.md` - CLI 使用纠正（需要更新）
- PTX ISA Guide - 官方参数规范

---

**结论**：PTX kernel 既支持按值传递（标量），也支持按引用传递（指针）。只有指针参数才需要先分配 device memory！
