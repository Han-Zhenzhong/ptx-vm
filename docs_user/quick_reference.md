# PTX 虚拟机 - 使用快速参考

**Authors**: Han-Zhenzhong, TongyiLingma, GitHub Copilot  
**Last Updated**: 2025-10-30

这是一个速查手册，提供 PTX 虚拟机的常用命令和 API 快速参考。

## 📑 目录

- [三种使用方式](#三种使用方式)
- [命令行选项](#命令行选项)
- [CLI 交互命令](#cli-交互命令)
- [API 函数参考](#api-函数参考)
- [常用工作流](#常用工作流)

---

## 三种使用方式

### 1. 🚀 直接执行模式（最快）
```bash
# 基本运行
./ptx_vm program.ptx

# 带日志级别
./ptx_vm --log-level debug program.ptx
./ptx_vm -l info program.ptx

# 查看帮助
./ptx_vm --help
```

**日志级别**：
- `debug` - 详细调试信息（显示寄存器、内存操作等）
- `info` - 一般信息（默认）
- `warning` - 警告和错误
- `error` - 仅错误

### 2. 💻 交互模式（用于调试）
```bash
# 启动交互模式
./ptx_vm

# 常用命令
> load program.ptx      # 加载程序
> alloc 1024            # 分配内存
> launch kernel 0x1000  # 启动内核
> memory 0x1000 16      # 查看内存
> dump                  # 显示统计
> quit                  # 退出
```

### 3. 🔧 API 模式（用于集成）
```cpp
#include "host_api.hpp"

HostAPI api;
api.initialize();
api.loadProgram("kernel.ptx");
// ... 使用 API 函数
```

---

## 命令行选项

```bash
./ptx_vm [选项] [ptx_file]

选项：
  -h, --help              显示帮助信息
  -l, --log-level LEVEL   设置日志级别 (debug|info|warning|error)
  
示例：
  ./ptx_vm program.ptx                      # 基本运行
  ./ptx_vm --log-level debug program.ptx   # 调试模式
  ./ptx_vm -l error program.ptx            # 仅显示错误
  ./ptx_vm                                  # 交互模式
```

---

## CLI 交互命令

### 程序加载
```bash
load <filename>         # 加载 PTX 文件
```

### 内存操作
```bash
alloc <size>                       # 分配内存
memory <address> [size]            # 查看内存
write <address> <value>            # 写入单个字节
fill <addr> <count> <v1> [v2]...   # 填充多个字节
memcpy <dest> <src> <size>         # 拷贝内存
loadfile <addr> <file> <size>      # 从文件加载
```

### 执行控制
```bash
run                     # 运行程序
step [count]            # 单步执行
launch <kernel> [params...]  # 启动内核
```

### 调试
```bash
break <address>         # 设置断点
watch <address>         # 设置监视点
register [all|predicate|pc]  # 查看寄存器
```

### 信息查看
```bash
dump                    # 显示执行统计
list                    # 显示反汇编
visualize <type>        # 可视化 (warp|memory|performance)
loglevel [level]        # 查看/设置日志级别
help [command]          # 帮助
```

### 其他
```bash
profile <file.csv>      # 开始性能分析
quit                    # 退出
```

---

## API 函数参考

### 初始化
```cpp
HostAPI hostAPI;
bool success = hostAPI.initialize();
```

### 内存管理
```cpp
// 分配内存
CUdeviceptr ptr;
hostAPI.cuMemAlloc(&ptr, size);

// 释放内存
hostAPI.cuMemFree(ptr);

// 主机到设备拷贝
hostAPI.cuMemcpyHtoD(devicePtr, hostPtr, size);

// 设备到主机拷贝
hostAPI.cuMemcpyDtoH(hostPtr, devicePtr, size);

// 设备到设备拷贝
hostAPI.cuMemcpyDtoD(destPtr, srcPtr, size);
```

### 程序加载
```cpp
// 加载 PTX 程序
hostAPI.loadProgram("kernel.ptx");

// 检查是否已加载
bool loaded = hostAPI.isProgramLoaded();
```

### 内核启动
```cpp
// 准备参数
void* params[] = { &ptr1, &ptr2, &value };

// 启动内核
hostAPI.cuLaunchKernel(
    kernel,              // 内核函数
    gridX, gridY, gridZ,   // 网格维度
    blockX, blockY, blockZ, // 块维度
    sharedMem,           // 共享内存大小
    stream,              // 流（可为 nullptr）
    params,              // 参数数组
    nullptr              // 额外参数
);
```

### 调试
```cpp
// 设置断点和监视点
hostAPI.setBreakpoint(address);
hostAPI.setWatchpoint(address);

// 查看信息
hostAPI.printRegisters();
hostAPI.printMemory(address, size);
hostAPI.printPerformanceCounters();
```

---

## 常用工作流

### 工作流 1：快速测试 PTX 文件
```bash
# 直接运行
./ptx_vm examples/simple_math_example.ptx

# 带调试信息
./ptx_vm --log-level debug examples/simple_math_example.ptx
```

### 工作流 2：交互式调试
```bash
$ ./ptx_vm
> load examples/control_flow_example.ptx
> alloc 1024
Allocated at: 0x10000
> fill 0x10000 8 1 2 3 4 5 6 7 8
> loglevel debug
> launch myKernel 0x10000
> memory 0x10000 16
> dump
> quit
```

### 工作流 3：API 编程（完整示例）
```cpp
#include "host_api.hpp"
#include <vector>

int main() {
    HostAPI api;
    api.initialize();
    
    // 分配和准备数据
    CUdeviceptr inPtr, outPtr;
    api.cuMemAlloc(&inPtr, 1024 * sizeof(int));
    api.cuMemAlloc(&outPtr, 1024 * sizeof(int));
    
    std::vector<int> data(1024);
    // ... 初始化 data
    api.cuMemcpyHtoD(inPtr, data.data(), 1024 * sizeof(int));
    
    // 加载和启动
    api.loadProgram("kernel.ptx");
    void* params[] = { &inPtr, &outPtr };
    api.cuLaunchKernel(kernel, 1,1,1, 32,1,1, 0, nullptr, params, nullptr);
    
    // 获取结果
    api.cuMemcpyDtoH(data.data(), outPtr, 1024 * sizeof(int));
    
    // 清理
    api.cuMemFree(inPtr);
    api.cuMemFree(outPtr);
    
    return 0;
}
```

### 工作流 4：性能分析
```bash
$ ./ptx_vm
> profile performance.csv
> load examples/comprehensive_test_suite.ptx
> launch testKernel 0x10000
> quit
$ cat performance.csv  # 查看性能数据
```

---

## 快速提示

### 💡 调试技巧
```bash
# 使用断点
> break 0x100
> run
# 程序在断点处停止

# 单步执行并查看状态
> step
> register all
> memory 0x10000 16
```

### 💡 内存初始化
```bash
# 方法 1: fill 命令
> alloc 256
> fill 0x10000 4 1 2 3 4

# 方法 2: 从文件加载
> loadfile 0x10000 data.bin 256

# 方法 3: API 中使用 cuMemcpyHtoD
```

### 💡 查看执行结果
```bash
# 交互模式
> memory 0x10000 32
> dump

# API 模式
hostAPI.printMemory(0x10000, 32);
hostAPI.printPerformanceCounters();
```

### 💡 日志控制
```bash
# 命令行
./ptx_vm -l debug program.ptx

# 交互模式
> loglevel debug
> run
> loglevel info  # 切换回普通模式
```

---

## 📚 更多信息

- 📖 [完整用户指南](./user_guide.md) - 详细使用说明
- 📖 [中文用户指南](./USER_GUIDE_CN.md) - 中文详细文档
- 📖 [API 文档](./api_documentation.md) - 完整 API 参考
- 📖 [日志系统](./logging_system.md) - 日志系统详细说明
- 📂 [示例代码](../examples/) - 示例程序
- 🐛 [问题报告](https://gitee.com/hanzhenzhong/ptx-vm/issues) - 提交问题

---

**提示**: 使用 `./ptx_vm --help` 或交互模式中的 `help` 命令查看更多信息。

## ✅ 已完成功能

### 1. 参数传递 ✅

```cpp
// Host 代码
float* d_data;
cuMemAlloc(&d_data, sizeof(float) * 100);
void* params[] = { &d_data, &size };
cuLaunchKernel(kernel, 1,1,1, 32,1,1, 0, 0, params, nullptr);

// PTX 代码
.entry kernel(.param .u64 data_ptr, .param .s32 size) {
    ld.param.u64 %rd1, [data_ptr];  // ✅ 现在可以正确读取！
}
```

---

### 2. 浮点寄存器 ✅

```cpp
// API
registerBank.writeFloatRegister(0, 3.14f);
float f = registerBank.readFloatRegister(0);

registerBank.writeDoubleRegister(1, 2.718);
double d = registerBank.readDoubleRegister(1);
```

```ptx
// PTX (待实现执行)
mov.f32 %f1, 3.14;
add.f32 %f2, %f1, %f3;  // 需要实现
```

---

### 3. 特殊寄存器 ✅

```cpp
// API
registerBank.setThreadId(5, 0, 0);
registerBank.setBlockId(2, 1, 0);
uint32_t tid = registerBank.readSpecialRegister(SpecialRegister::TID_X);
```

```ptx
// PTX (待实现执行)
mov.u32 %r1, %tid.x;     // 需要实现
mov.u32 %r2, %ctaid.x;   // 需要实现
```

---

### 4. 指令类型定义 ✅

```cpp
// 新增 36 个指令类型
InstructionTypes::ADD_F32
InstructionTypes::SETP
InstructionTypes::CVT
InstructionTypes::ATOM_ADD
// ... 等

// 新增枚举
CompareOp::LT, CompareOp::EQ
DataType::F32, DataType::S32
```

---

## 🚧 待实现（有完整代码示例）

### 5. 浮点指令执行 🚧

**参考**: `new_features_implementation_guide.md` 第5节

```cpp
// 需要在 executor.cpp 添加
bool executeADD_F32(const DecodedInstruction& instr) {
    float src1 = m_registerBank->readFloatRegister(instr.sources[0].registerIndex);
    float src2 = m_registerBank->readFloatRegister(instr.sources[1].registerIndex);
    float result = src1 + src2;
    m_registerBank->writeFloatRegister(instr.dest.registerIndex, result);
    return true;
}

// 在 parser.cpp 添加识别
if (opcode == "add" && hasModifier(".f32")) 
    return InstructionTypes::ADD_F32;
```

---

### 6. SETP 比较指令 🚧

**参考**: `new_features_implementation_guide.md` 第6节

```cpp
bool executeSETP(const DecodedInstruction& instr) {
    int32_t src1 = m_registerBank->readRegister(instr.sources[0].registerIndex);
    int32_t src2 = m_registerBank->readRegister(instr.sources[1].registerIndex);
    
    bool result = false;
    switch (instr.compareOp) {
        case CompareOp::LT: result = (src1 < src2); break;
        case CompareOp::EQ: result = (src1 == src2); break;
        // ...
    }
    
    m_registerBank->writePredicate(instr.dest.predicateIndex, result);
    return true;
}
```

```ptx
setp.lt.s32 %p1, %r1, %r2;  // %p1 = (%r1 < %r2)
@%p1 bra TARGET;             // 条件分支
```

---

### 7. SELP 条件选择 🚧

**参考**: `new_features_implementation_guide.md` 第7节

```cpp
bool executeSELP(const DecodedInstruction& instr) {
    bool pred = m_registerBank->readPredicate(instr.sources[2].predicateIndex);
    uint64_t src1 = m_registerBank->readRegister(instr.sources[0].registerIndex);
    uint64_t src2 = m_registerBank->readRegister(instr.sources[1].registerIndex);
    uint64_t result = pred ? src1 : src2;
    m_registerBank->writeRegister(instr.dest.registerIndex, result);
    return true;
}
```

```ptx
selp.s32 %r3, %r1, %r2, %p1;  // %r3 = %p1 ? %r1 : %r2
```

---

### 8. CVT 类型转换 🚧

**参考**: `new_features_implementation_guide.md` 第8节

```cpp
bool executeCVT(const DecodedInstruction& instr) {
    if (instr.srcType == DataType::F32 && instr.dstType == DataType::S32) {
        float src = m_registerBank->readFloatRegister(instr.sources[0].registerIndex);
        int32_t dst = static_cast<int32_t>(src);
        m_registerBank->writeRegister(instr.dest.registerIndex, dst);
    }
    // ... 其他转换
    return true;
}
```

```ptx
cvt.s32.f32 %r1, %f1;  // %r1 = (int)%f1
cvt.f32.s32 %f1, %r1;  // %f1 = (float)%r1
```

---

### 9. 原子操作 🚧

**参考**: `new_features_implementation_guide.md` 第9节

```cpp
bool executeATOM_ADD(const DecodedInstruction& instr) {
    uint64_t address = instr.sources[0].address;
    uint32_t addValue = m_registerBank->readRegister(instr.sources[1].registerIndex);
    
    uint32_t oldValue = m_memorySubsystem->read<uint32_t>(
        instr.memorySpace, address);
    uint32_t newValue = oldValue + addValue;
    m_memorySubsystem->write<uint32_t>(
        instr.memorySpace, address, newValue);
    
    m_registerBank->writeRegister(instr.dest.registerIndex, oldValue);
    return true;
}
```

```ptx
atom.global.add.u32 %r1, [%rd1], %r2;  // 原子加法
atom.global.cas.b32 %r1, [%rd1], %r2, %r3;  // 比较交换
```

---

## 📚 完整文档

| 文档 | 用途 |
|------|------|
| [comprehensive_implementation_analysis.md](../docs_dev/comprehensive_implementation_analysis.md) | 完整分析和问题诊断 |
| [new_features_implementation_guide.md](../docs_dev/new_features_implementation_guide.md) | **详细代码示例和实现步骤** ⭐ |
| [implementation_summary_phase1.md](../docs_dev/archive/implementation_summary_phase1.md) | 阶段1实现总结 |
| 本文档 | 快速参考 |

---

## 🎯 实现优先级

### 第1优先级（立即）✅
- ✅ 参数传递
- ✅ 浮点寄存器
- ✅ 特殊寄存器
- ✅ 指令类型定义

### 第2优先级（本周）🚧
- 🚧 ADD_F32, MUL_F32, DIV_F32
- 🚧 SETP 指令

### 第3优先级（下周）🚧
- 🚧 SELP, CVT
- 🚧 MOV 特殊寄存器

### 第4优先级（两周内）🚧
- 🚧 原子操作
- 🚧 高级浮点（FMA, SQRT）

---

## 🧪 测试文件

所有测试示例在 `new_features_implementation_guide.md` 第10节：

- 测试1: 浮点运算
- 测试2: 比较和分支
- 测试3: 类型转换
- 测试4: 特殊寄存器
- 测试5: 原子操作

---

**快速开始**: 查看 `new_features_implementation_guide.md` 获取完整代码！
