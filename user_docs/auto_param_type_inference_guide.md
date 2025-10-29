# PTX 参数自动类型推断 - 使用指南

**Authors**: Han-Zhenzhong, TongyiLingma, GitHub Copilot  
**Last Updated**: 2025-10-29

## 🎯 概述

PTX VM CLI 现在支持**自动参数类型推断**！

根据 PTX kernel 的函数签名，CLI 会自动将命令行参数转换为正确的类型：
- **指针参数** (`.u64`) → 解析为设备内存地址
- **标量参数** (`.u32`, `.f32`, `.s32` 等) → 解析为直接数值

**不需要手动指定参数类型！**

---

## 🔥 核心特性

### 1. 自动类型检测

CLI 会：
1. 读取已加载的 PTX 程序
2. 查找指定的 kernel 函数
3. 提取参数列表及其类型（包括无参数情况）
4. 根据参数类型自动转换命令行输入

### 2. 支持的参数类型

| PTX 类型 | C++ 类型 | CLI 输入示例 | 用途 |
|----------|----------|------------|------|
| **无参数** | - | `launch kernel` | 无需任何参数 |
| `.u8` | `uint8_t` | `255` | 小整数 |
| `.u16` | `uint16_t` | `65535` | 短整数 |
| `.u32` | `uint32_t` | `1024` | 无符号整数 |
| `.s32` | `int32_t` | `-100` | 有符号整数 |
| `.u64` | `uint64_t` | `0x10000` | 指针或大整数 |
| `.s64` | `int64_t` | `-1000000` | 有符号长整数 |
| `.f32` | `float` | `2.5` | 单精度浮点 |
| `.f64` | `double` | `3.14159` | 双精度浮点 |

---

## 📖 使用示例

### 示例 0：无参数 kernel

**PTX 签名**：
```ptx
.visible .entry noParamKernel()
```

**CLI 使用**：
```bash
ptx-vm> load examples/no_param_kernels.ptx
Program loaded successfully.

# 直接启动，不需要任何参数或内存分配
ptx-vm> launch noParamKernel

Launching kernel with no parameters

Launching kernel: noParamKernel
Grid dimensions: 1 x 1 x 1
Block dimensions: 32 x 1 x 1

✓ Kernel launched successfully

# Kernel 可能会访问固定地址的内存或使用特殊寄存器
ptx-vm> memory 0x10000 16
0x10000: 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
```

**关键点**：
- ✅ 不需要 `alloc` 任何内存
- ✅ 不需要传递任何参数
- ✅ Kernel 可以使用特殊寄存器（`%tid.x`, `%ctaid.x` 等）
- ✅ Kernel 可以访问固定地址的全局内存

**常见用途**：
- 调试和测试
- 初始化固定内存区域
- 使用特殊寄存器的计算
- Atomic 操作到固定地址

---

### 示例 1：纯指针参数（向量加法）

**PTX 签名**：
```ptx
.visible .entry vecAdd(
    .param .u64 A,
    .param .u64 B,
    .param .u64 C
)
```

**CLI 使用**：
```bash
ptx-vm> load examples/vecAdd.ptx
Program loaded successfully.

# 为三个数组分配内存
ptx-vm> alloc 32
Allocated 32 bytes at address 0x10000

ptx-vm> alloc 32
Allocated 32 bytes at address 0x10020

ptx-vm> alloc 32
Allocated 32 bytes at address 0x10040

# 填充输入数据
ptx-vm> fill 0x10000 8 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0
Filled 8 float values

ptx-vm> fill 0x10020 8 10.0 20.0 30.0 40.0 50.0 60.0 70.0 80.0
Filled 8 float values

# 启动 kernel - 参数自动识别为指针
ptx-vm> launch vecAdd 0x10000 0x10020 0x10040

Parsing kernel parameters:
  [0] A (.u64): device address 0x10000
  [1] B (.u64): device address 0x10020
  [2] C (.u64): device address 0x10040

Launching kernel: vecAdd
Grid dimensions: 1 x 1 x 1
Block dimensions: 32 x 1 x 1

✓ Kernel launched successfully
```

---

### 示例 2：混合参数（指针 + 标量）

**PTX 签名**：
```ptx
.visible .entry scaleArray(
    .param .u64 data_ptr,   // 指针：指向数组
    .param .u32 N,          // 标量：数组大小
    .param .f32 scale       // 标量：缩放因子
)
```

**CLI 使用**：
```bash
ptx-vm> load examples/mixed_param_test.ptx
Program loaded successfully.

# 只为指针参数分配内存
ptx-vm> alloc 4096
Allocated 4096 bytes at address 0x10000

# 填充数据（1024个浮点数）
ptx-vm> fill 0x10000 1024 1.0 1.0 1.0 ...
Filled 1024 float values

# 启动 kernel - 自动识别类型
ptx-vm> launch scaleArray 0x10000 1024 2.5
                         ↑       ↑    ↑
                      pointer  u32  f32

Parsing kernel parameters:
  [0] data_ptr (.u64): device address 0x10000
  [1] N (.u32): value 1024
  [2] scale (.f32): value 2.5

Launching kernel: scaleArray
✓ Kernel launched successfully

# 查看结果（所有值都乘以 2.5）
ptx-vm> memory 0x10000 16
0x10000: 2.5 2.5 2.5 2.5 ...
```

**关键点**：
- ✅ `0x10000` 被识别为 `.u64` 指针
- ✅ `1024` 被识别为 `.u32` 标量
- ✅ `2.5` 被识别为 `.f32` 标量
- ✅ 标量参数不需要 `alloc`！

---

### 示例 3：纯标量参数（无需内存分配）

**PTX 签名**：
```ptx
.visible .entry computeScalars(
    .param .u32 a,
    .param .u32 b,
    .param .f32 c
)
```

**CLI 使用**：
```bash
ptx-vm> load examples/mixed_param_test.ptx
Program loaded successfully.

# 直接启动，不需要任何 alloc！
ptx-vm> launch computeScalars 100 200 1.5

Parsing kernel parameters:
  [0] a (.u32): value 100
  [1] b (.u32): value 200
  [2] c (.f32): value 1.5

Launching kernel: computeScalars
✓ Kernel launched successfully

# 结果存储在固定地址 0x10000
# (100 + 200) * 1.5 = 450.0
ptx-vm> memory 0x10000 4
0x10000: 450.0
```

---

### 示例 4：复杂混合参数

**PTX 签名**：
```ptx
.visible .entry complexKernel(
    .param .u64 input_ptr,
    .param .u64 output_ptr,
    .param .u32 N,
    .param .f32 alpha,
    .param .s32 beta,
    .param .f64 gamma
)
```

**CLI 使用**：
```bash
ptx-vm> alloc 4096
Allocated 4096 bytes at address 0x10000

ptx-vm> alloc 4096
Allocated 4096 bytes at address 0x20000

ptx-vm> fill 0x10000 1024 1.0 2.0 3.0 ...

ptx-vm> launch complexKernel 0x10000 0x20000 1024 2.5 -10 3.14159

Parsing kernel parameters:
  [0] input_ptr (.u64): device address 0x10000
  [1] output_ptr (.u64): device address 0x20000
  [2] N (.u32): value 1024
  [3] alpha (.f32): value 2.5
  [4] beta (.s32): value -10
  [5] gamma (.f64): value 3.14159

✓ Kernel launched successfully
```

---

## 🔍 参数类型推断规则

### 指针参数 (`.u64` 或 `.s64`)

**条件**：`param.isPointer == true`

**解析方式**：
- 作为 64 位地址
- 支持十六进制 (`0x10000`) 和十进制 (`65536`)
- 必须指向已分配的设备内存

**示例**：
```bash
launch kernel 0x10000  # 十六进制
launch kernel 65536    # 十进制（同样的地址）
```

### 标量参数

#### 整数类型

| 类型 | 范围 | 示例 |
|------|------|------|
| `.u8` | 0 ~ 255 | `255` |
| `.s8` | -128 ~ 127 | `-100` |
| `.u16` | 0 ~ 65535 | `1000` |
| `.s16` | -32768 ~ 32767 | `-1000` |
| `.u32` | 0 ~ 4294967295 | `1024` |
| `.s32` | -2147483648 ~ 2147483647 | `-1024` |
| `.u64` (非指针) | 0 ~ 2^64-1 | `1000000000000` |
| `.s64` | -2^63 ~ 2^63-1 | `-1000000000000` |

#### 浮点类型

| 类型 | 精度 | 示例 |
|------|------|------|
| `.f32` | 单精度（32位） | `2.5`, `3.14`, `-1.0` |
| `.f64` | 双精度（64位） | `3.14159265359`, `2.718281828` |

---

## ⚙️ 错误处理

### 1. 参数数量不匹配

```bash
ptx-vm> launch scaleArray 0x10000 1024
# 错误：缺少 scale 参数

Error: Parameter count mismatch: expected 3, got 2

Kernel signature: scaleArray(
  [0] .u64 data_ptr (pointer - needs device address)
  [1] .u32 N (scalar - needs value)
  [2] .f32 scale (scalar - needs value)
)
```

### 2. 参数类型转换失败

```bash
ptx-vm> launch scaleArray 0x10000 abc 2.5
# 错误：'abc' 不是有效的 u32 值

Error: Failed to parse parameter 1 ('N' of type .u32) from value: abc
```

### 3. Kernel 不存在

```bash
ptx-vm> launch nonExistentKernel 0x10000

Error: Kernel 'nonExistentKernel' not found in loaded PTX program.
Available kernels:
  - scaleArray
  - addOffset
  - computeScalars
  - complexKernel
```

### 4. 未加载 PTX 程序

```bash
ptx-vm> launch kernel 0x10000

Error: No PTX program loaded. Use 'load' first.
```

---

## 📚 完整工作流程

### 场景：数组缩放（混合参数）

**目标**：将数组中的每个元素乘以 2.5

**步骤**：

```bash
# 1. 启动 PTX VM
$ ./ptx_vm

# 2. 加载 PTX 程序
ptx-vm> load examples/mixed_param_test.ptx
Program loaded successfully.

# 3. 查看可用的 kernels
ptx-vm> help launch
# （会显示所有可用的 kernel）

# 4. 为数组数据分配内存
ptx-vm> alloc 4096
Allocated 4096 bytes at address 0x10000

# 5. 填充初始数据（1024个浮点数，都是 1.0）
ptx-vm> fill 0x10000 1024 1.0
Filled 1024 float values (all 1.0)

# 6. 启动 kernel（自动类型推断）
ptx-vm> launch scaleArray 0x10000 1024 2.5

Parsing kernel parameters:
  [0] data_ptr (.u64): device address 0x10000
  [1] N (.u32): value 1024
  [2] scale (.f32): value 2.5

✓ Kernel launched successfully

# 7. 验证结果
ptx-vm> memory 0x10000 16
0x10000: 2.5 2.5 2.5 2.5 ...

# 8. 再次缩放（2.5 * 3.0 = 7.5）
ptx-vm> launch scaleArray 0x10000 1024 3.0
✓ Kernel launched successfully

ptx-vm> memory 0x10000 16
0x10000: 7.5 7.5 7.5 7.5 ...
```

---

## 🆚 与之前的对比

### 之前的错误理解

```bash
# ❌ 错误：认为所有参数都必须是设备地址
ptx-vm> alloc 4
0x10000
ptx-vm> fill 0x10000 1 0x00 0x04 0x00 0x00  # 存储 N=1024
ptx-vm> alloc 4
0x10004
ptx-vm> fill 0x10004 1 0x00 0x00 0x20 0x40  # 存储 scale=2.5
ptx-vm> launch kernel 0x10000 0x10004  # ❌ 错误！
```

### 现在的正确方式

**场景 1：无参数 kernel**
```bash
# ✅ 正确：无需任何准备
ptx-vm> launch testKernel
✓ Kernel launched successfully
```

**场景 2：混合参数 kernel**
```bash
# ✅ 正确：自动区分指针和标量
ptx-vm> alloc 4096    # 只为指针参数分配
0x10000
ptx-vm> fill 0x10000 1024 1.0 1.0 ...
ptx-vm> launch kernel 0x10000 1024 2.5  # ✅ 自动识别类型
```

---

## 💡 最佳实践

### 1. 查看 Kernel 签名

在启动 kernel 前，确保知道参数类型：

```bash
ptx-vm> launch scaleArray
# 会显示参数列表和类型要求
```

### 2. 使用十六进制表示地址

```bash
# 推荐
launch kernel 0x10000 1024 2.5

# 也可以，但不太直观
launch kernel 65536 1024 2.5
```

### 3. 浮点数必须包含小数点

```bash
# ✅ 正确
launch kernel 0x10000 1024 2.5

# ⚠️ 可能被解析为整数
launch kernel 0x10000 1024 2
# 应该写成 2.0
```

### 4. 标量参数不需要 alloc

```bash
# ❌ 不必要的内存分配
ptx-vm> alloc 4    # 不需要！
ptx-vm> fill 0x10000 1 1024
ptx-vm> launch kernel 0x10000 0x10000 2.5

# ✅ 直接传值
ptx-vm> alloc 4096  # 只为指针参数分配
ptx-vm> launch kernel 0x10000 1024 2.5
```

---

## 🔧 技术实现

### CLI 内部流程

1. **解析命令**：
   ```cpp
   launch scaleArray 0x10000 1024 2.5
   ```

2. **查找 Kernel**：
   ```cpp
   const PTXFunction* kernel = findKernel("scaleArray");
   // kernel->parameters = [
   //   {name: "data_ptr", type: ".u64", isPointer: true},
   //   {name: "N", type: ".u32", isPointer: false},
   //   {name: "scale", type: ".f32", isPointer: false}
   // ]
   ```

3. **参数转换**：
   ```cpp
   parseParameterValue("0x10000", param[0])  // → uint64_t = 0x10000
   parseParameterValue("1024", param[1])     // → uint32_t = 1024
   parseParameterValue("2.5", param[2])      // → float = 2.5
   ```

4. **复制到参数内存**：
   ```cpp
   // 参数内存布局（基址 0x1000）：
   // offset 0:  0x0000000000010000  (data_ptr: 8 bytes)
   // offset 8:  0x00000400          (N: 4 bytes)
   // offset 12: 0x40200000          (scale: 4 bytes)
   ```

5. **启动 Kernel**：
   ```cpp
   cuLaunchKernel(..., kernelParams, ...)
   ```

---

## 📖 参考文档

- `docs/param_type_of_ptx_entry_function.md` - PTX 参数类型详解
- `docs/ptx_entry_function_complete_guide.md` - 完整参数传递指南
- `docs/how_CudaC_and_PTX_called_by_HostC.md` - CUDA 调用模型
- `examples/mixed_param_test.ptx` - 混合参数示例
- `examples/parameter_passing_example.cpp` - Host API 使用示例

---

## ✅ 总结

### 关键要点

1. **自动类型推断**：CLI 根据 PTX 签名自动转换参数
2. **指针 vs 标量**：只有指针参数需要 `alloc`
3. **简化工作流**：不需要手动管理标量参数的内存
4. **错误提示清晰**：参数数量/类型不匹配时会显示详细信息

### 支持的场景

| 场景 | 示例 | 需要 alloc |
|------|------|----------|
| 无参数 | `launch testKernel` | ❌ 否 |
| 纯指针 | `launch vecAdd 0x10000 0x10020 0x10040` | ✅ 是 |
| 纯标量 | `launch compute 100 200 1.5` | ❌ 否 |
| 混合 | `launch scaleArray 0x10000 1024 2.5` | ✅ 部分（只为指针） |

---

**现在开始使用自动类型推断，简化您的 PTX 开发体验！** 🚀
