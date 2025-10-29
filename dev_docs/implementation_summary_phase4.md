# PTX VM 实现总结 - 第4阶段：原子操作

## 概述

本文档总结了按照 `new_features_implementation_guide.md` 第9节实现的原子操作功能。

**实现日期**: 2025年1月

**实现范围**: 
- 6种基本原子操作
- GLOBAL 和 SHARED 内存空间支持
- 完整的读-修改-写语义
- 原子操作返回旧值

## 功能列表

### 已实现的原子操作

| 操作 | 指令 | 功能描述 | 返回值 |
|------|------|---------|--------|
| ATOM_ADD | atom.global.add.u32 | 原子加法 | 修改前的值 |
| ATOM_SUB | atom.global.sub.u32 | 原子减法 | 修改前的值 |
| ATOM_EXCH | atom.global.exch.u32 | 原子交换 | 修改前的值 |
| ATOM_CAS | atom.global.cas.u32 | 原子比较并交换 | 修改前的值 |
| ATOM_MIN | atom.global.min.u32 | 原子最小值 | 修改前的值 |
| ATOM_MAX | atom.global.max.u32 | 原子最大值 | 修改前的值 |

## 实现细节

### 1. 原子加法 (ATOM_ADD)

**功能**: 原子地执行 `memory[address] = memory[address] + value`

**PTX 语法**:
```ptx
atom.global.add.u32 d, [a], b;
```

**实现代码**:
```cpp
bool Executor::executeATOM_ADD(const DecodedInstruction& instr) {
    // 从操作数获取内存地址
    uint64_t address;
    if (instr.sources[0].type == OperandType::MEMORY) {
        address = instr.sources[0].address;
    } else if (instr.sources[0].type == OperandType::REGISTER) {
        address = m_registerBank->readRegister(instr.sources[0].registerIndex);
    } else {
        return false;
    }
    
    // 读取加数
    uint32_t addValue = static_cast<uint32_t>(
        m_registerBank->readRegister(instr.sources[1].registerIndex));
    
    // 确定内存空间
    MemorySpace space = (instr.memorySpace != MemorySpace::GLOBAL && 
                        instr.memorySpace != MemorySpace::SHARED) 
                        ? MemorySpace::GLOBAL : instr.memorySpace;
    
    // 读取旧值
    uint32_t oldValue = m_memorySubsystem->read<uint32_t>(space, address);
    
    // 计算新值并写回
    uint32_t newValue = oldValue + addValue;
    m_memorySubsystem->write<uint32_t>(space, address, newValue);
    
    // 返回旧值到目标寄存器
    if (instr.dest.type == OperandType::REGISTER) {
        m_registerBank->writeRegister(instr.dest.registerIndex, 
                                      static_cast<uint64_t>(oldValue));
    }
    
    return true;
}
```

**使用示例**:
```ptx
// 原子递增计数器
mov.u32 %r1, 1;
atom.global.add.u32 %r2, [counter_addr], %r1;
// %r2 现在包含递增前的值
```

### 2. 原子减法 (ATOM_SUB)

**功能**: 原子地执行 `memory[address] = memory[address] - value`

**PTX 语法**:
```ptx
atom.global.sub.u32 d, [a], b;
```

**实现特点**:
- 与 ATOM_ADD 类似的结构
- 执行减法操作: `newValue = oldValue - subValue`
- 返回修改前的值

**使用示例**:
```ptx
// 原子递减计数器
mov.u32 %r1, 1;
atom.global.sub.u32 %r2, [counter_addr], %r1;
```

### 3. 原子交换 (ATOM_EXCH)

**功能**: 原子地执行 `memory[address] = value`，返回旧值

**PTX 语法**:
```ptx
atom.global.exch.u32 d, [a], b;
```

**实现代码**:
```cpp
bool Executor::executeATOM_EXCH(const DecodedInstruction& instr) {
    // 获取地址和新值
    uint64_t address = getMemoryAddress(instr.sources[0]);
    uint32_t newValue = static_cast<uint32_t>(
        m_registerBank->readRegister(instr.sources[1].registerIndex));
    
    MemorySpace space = determineMemorySpace(instr);
    
    // 读取旧值
    uint32_t oldValue = m_memorySubsystem->read<uint32_t>(space, address);
    
    // 写入新值
    m_memorySubsystem->write<uint32_t>(space, address, newValue);
    
    // 返回旧值
    if (instr.dest.type == OperandType::REGISTER) {
        m_registerBank->writeRegister(instr.dest.registerIndex, 
                                      static_cast<uint64_t>(oldValue));
    }
    
    return true;
}
```

**使用场景**:
- 实现自旋锁
- 快速交换数据
- 无锁数据结构

**使用示例**:
```ptx
// 获取锁（将 1 写入 lock_addr）
mov.u32 %r1, 1;
atom.global.exch.u32 %r2, [lock_addr], %r1;
// %r2 == 0 表示获取到锁
```

### 4. 原子比较并交换 (ATOM_CAS)

**功能**: 如果 `memory[address] == compare`，则 `memory[address] = value`

**PTX 语法**:
```ptx
atom.global.cas.u32 d, [a], compare, new;
```

**实现代码**:
```cpp
bool Executor::executeATOM_CAS(const DecodedInstruction& instr) {
    // 获取地址、比较值、新值
    uint64_t address = getMemoryAddress(instr.sources[0]);
    uint32_t compareValue = static_cast<uint32_t>(
        m_registerBank->readRegister(instr.sources[1].registerIndex));
    uint32_t newValue = static_cast<uint32_t>(
        m_registerBank->readRegister(instr.sources[2].registerIndex));
    
    MemorySpace space = determineMemorySpace(instr);
    
    // 读取旧值
    uint32_t oldValue = m_memorySubsystem->read<uint32_t>(space, address);
    
    // 仅当旧值等于比较值时才写入新值
    if (oldValue == compareValue) {
        m_memorySubsystem->write<uint32_t>(space, address, newValue);
    }
    
    // 始终返回旧值
    if (instr.dest.type == OperandType::REGISTER) {
        m_registerBank->writeRegister(instr.dest.registerIndex, 
                                      static_cast<uint64_t>(oldValue));
    }
    
    return true;
}
```

**使用场景**:
- 无锁算法
- ABA 问题解决
- 条件更新

**使用示例**:
```ptx
// 尝试从 0 更新到 1（获取锁）
mov.u32 %r1, 0;    // 期望值
mov.u32 %r2, 1;    // 新值
atom.global.cas.u32 %r3, [lock_addr], %r1, %r2;
setp.eq.u32 %p1, %r3, 0;  // 检查是否成功
@%p1 bra locked;
```

### 5. 原子最小值 (ATOM_MIN)

**功能**: 原子地执行 `memory[address] = min(memory[address], value)`

**PTX 语法**:
```ptx
atom.global.min.u32 d, [a], b;
```

**实现代码**:
```cpp
bool Executor::executeATOM_MIN(const DecodedInstruction& instr) {
    uint64_t address = getMemoryAddress(instr.sources[0]);
    uint32_t compareValue = static_cast<uint32_t>(
        m_registerBank->readRegister(instr.sources[1].registerIndex));
    
    MemorySpace space = determineMemorySpace(instr);
    uint32_t oldValue = m_memorySubsystem->read<uint32_t>(space, address);
    
    // 计算最小值
    uint32_t newValue = (oldValue < compareValue) ? oldValue : compareValue;
    m_memorySubsystem->write<uint32_t>(space, address, newValue);
    
    // 返回旧值
    if (instr.dest.type == OperandType::REGISTER) {
        m_registerBank->writeRegister(instr.dest.registerIndex, 
                                      static_cast<uint64_t>(oldValue));
    }
    
    return true;
}
```

**使用示例**:
```ptx
// 查找数组中的最小值
ld.global.u32 %r1, [data_addr];
atom.global.min.u32 %r2, [global_min], %r1;
```

### 6. 原子最大值 (ATOM_MAX)

**功能**: 原子地执行 `memory[address] = max(memory[address], value)`

**PTX 语法**:
```ptx
atom.global.max.u32 d, [a], b;
```

**实现特点**:
- 与 ATOM_MIN 类似
- 执行最大值操作: `newValue = max(oldValue, compareValue)`
- 返回修改前的值

**使用示例**:
```ptx
// 查找数组中的最大值
ld.global.u32 %r1, [data_addr];
atom.global.max.u32 %r2, [global_max], %r1;
```

## 内存空间支持

### 支持的内存空间

| 内存空间 | 描述 | PTX 语法 |
|---------|------|---------|
| GLOBAL | 全局内存 | atom.global.op |
| SHARED | 共享内存 | atom.shared.op |

### 内存空间处理

```cpp
MemorySpace determineMemorySpace(const DecodedInstruction& instr) {
    // 如果指令中指定了有效的内存空间，使用它
    if (instr.memorySpace == MemorySpace::GLOBAL || 
        instr.memorySpace == MemorySpace::SHARED) {
        return instr.memorySpace;
    }
    // 否则默认为 GLOBAL
    return MemorySpace::GLOBAL;
}
```

## 地址操作数处理

### 两种地址形式

1. **直接内存地址**:
```ptx
atom.global.add.u32 %r1, [0x1000], %r2;  // 直接地址
```

2. **间接寄存器地址**:
```ptx
mov.u64 %rd1, 0x1000;
atom.global.add.u32 %r1, [%rd1], %r2;    // 寄存器中的地址
```

### 实现代码

```cpp
uint64_t getMemoryAddress(const Operand& operand) {
    if (operand.type == OperandType::MEMORY) {
        // 直接内存地址
        return operand.address;
    } else if (operand.type == OperandType::REGISTER) {
        // 从寄存器读取地址
        return m_registerBank->readRegister(operand.registerIndex);
    }
    return 0;  // 错误情况
}
```

## 完整测试示例

### 测试所有原子操作

```ptx
.entry test_atomic_operations(
    .param .u64 counter_ptr
)
{
    .reg .u32 %r<10>;
    .reg .u64 %rd1;
    
    ld.param.u64 %rd1, [counter_ptr];
    
    // Test ATOM.ADD
    mov.u32 %r1, 5;
    atom.global.add.u32 %r2, [%rd1], %r1;
    
    // Test ATOM.SUB
    mov.u32 %r3, 3;
    atom.global.sub.u32 %r4, [%rd1+4], %r3;
    
    // Test ATOM.EXCH
    mov.u32 %r5, 999;
    atom.global.exch.u32 %r6, [%rd1+8], %r5;
    
    // Test ATOM.CAS
    mov.u32 %r7, 100;  // Compare value
    mov.u32 %r8, 200;  // New value
    atom.global.cas.u32 %r9, [%rd1+12], %r7, %r8;
    
    // Store old values
    st.global.u32 [%rd1+16], %r2;
    st.global.u32 [%rd1+20], %r4;
    st.global.u32 [%rd1+24], %r6;
    st.global.u32 [%rd1+28], %r9;
    
    exit;
}
```

### 测试 MIN/MAX

```ptx
.entry test_atomic_min_max(
    .param .u64 data_ptr
)
{
    .reg .u32 %r<10>;
    .reg .u64 %rd1;
    
    ld.param.u64 %rd1, [data_ptr];
    
    // Test ATOM.MIN
    mov.u32 %r1, 50;
    atom.global.min.u32 %r2, [%rd1], %r1;
    
    // Test ATOM.MAX
    mov.u32 %r3, 100;
    atom.global.max.u32 %r4, [%rd1+4], %r3;
    
    // Store old values
    st.global.u32 [%rd1+8], %r2;
    st.global.u32 [%rd1+12], %r4;
    
    exit;
}
```

## 性能考虑

### 1. 原子操作开销

- **读取**: 1次内存读取
- **计算**: 简单的算术或比较操作
- **写入**: 1次内存写入
- **总计**: 约 2-3 倍于普通内存操作

### 2. 冲突处理

当前实现是简化版本：
- 不处理实际的原子性冲突
- 适用于单线程测试
- 多线程环境需要额外的同步机制

### 3. 优化建议

对于未来的多线程支持：
```cpp
// 伪代码：使用互斥锁保证原子性
std::lock_guard<std::mutex> lock(m_memoryMutex);
uint32_t oldValue = read(address);
uint32_t newValue = oldValue + addValue;
write(address, newValue);
```

## 与 CUDA 的对应关系

| PTX 原子操作 | CUDA 原子函数 | 功能 |
|-------------|--------------|------|
| atom.add | atomicAdd | 原子加法 |
| atom.sub | atomicSub | 原子减法 |
| atom.exch | atomicExch | 原子交换 |
| atom.cas | atomicCAS | 原子比较并交换 |
| atom.min | atomicMin | 原子最小值 |
| atom.max | atomicMax | 原子最大值 |

## 代码变更总结

### 修改的文件

1. **src/execution/executor.cpp** (~250 行新增)
   - 添加 6 个原子操作执行函数
   - 添加 6 个 switch case 分支
   - 实现完整的读-修改-写语义

### 代码统计

- 新增函数: 6 个
- 新增代码行: ~250 行
- 修改的 switch case: 6 个
- 测试示例: 2 个完整的 PTX 测试函数

## 测试验证

### 单元测试需求

1. **ATOM_ADD 测试**
   - 初始值: 10
   - 添加: 5
   - 期望新值: 15
   - 期望返回值: 10

2. **ATOM_SUB 测试**
   - 初始值: 20
   - 减去: 7
   - 期望新值: 13
   - 期望返回值: 20

3. **ATOM_CAS 测试**
   - 初始值: 100
   - 比较值: 100 (匹配)
   - 新值: 200
   - 期望新值: 200
   - 期望返回值: 100

4. **ATOM_CAS 失败测试**
   - 初始值: 100
   - 比较值: 50 (不匹配)
   - 新值: 200
   - 期望新值: 100 (不变)
   - 期望返回值: 100

### 集成测试

使用 `comprehensive_test_suite.ptx` 中的测试函数：
- `test_atomic_operations`: 测试 ADD, SUB, EXCH, CAS
- `test_atomic_min_max`: 测试 MIN, MAX

## 已知限制

1. **数据类型支持**
   - 当前仅支持 u32 (32位无符号整数)
   - PTX 还支持 s32, u64, s64, f32, f64

2. **内存空间**
   - 支持 GLOBAL 和 SHARED
   - 不支持 LOCAL 和 TEXTURE 内存

3. **原子性**
   - 单线程环境完全正确
   - 多线程环境需要额外的互斥机制

4. **性能**
   - 未优化的顺序执行
   - 无硬件级别的原子操作支持

## 未来扩展

### 1. 更多原子操作

```cpp
// 待实现的原子操作
ATOM_INC,    // 原子递增（带环绕）
ATOM_DEC,    // 原子递减（带环绕）
ATOM_AND,    // 原子按位与
ATOM_OR,     // 原子按位或
ATOM_XOR,    // 原子按位异或
```

### 2. 更多数据类型

```ptx
atom.global.add.s32 d, [a], b;  // 有符号 32 位
atom.global.add.u64 d, [a], b;  // 无符号 64 位
atom.global.add.f32 d, [a], b;  // 单精度浮点
```

### 3. 更多内存空间

```ptx
atom.local.add.u32 d, [a], b;   // 本地内存
```

### 4. 作用域修饰符

```ptx
atom.global.gpu.add.u32 d, [a], b;     // GPU 作用域
atom.global.cta.add.u32 d, [a], b;     // CTA 作用域
atom.global.sys.add.u32 d, [a], b;     // 系统作用域
```

## 总结

### 实现成果

✅ **6 种基本原子操作**
- ATOM_ADD: 原子加法
- ATOM_SUB: 原子减法
- ATOM_EXCH: 原子交换
- ATOM_CAS: 原子比较并交换
- ATOM_MIN: 原子最小值
- ATOM_MAX: 原子最大值

✅ **完整的功能支持**
- 读-修改-写语义
- 旧值返回
- GLOBAL 和 SHARED 内存空间
- 直接和间接地址

✅ **测试覆盖**
- comprehensive_test_suite.ptx 包含完整测试
- 预期结果清晰定义

### 下一步工作

1. 实现剩余的原子操作（INC, DEC, AND, OR, XOR）
2. 添加更多数据类型支持
3. 实现多线程原子性保证
4. 性能优化和基准测试
5. 完整的单元测试和集成测试

### 与其他阶段的集成

- **第1阶段**: 使用参数传递和寄存器系统
- **第2阶段**: 可与浮点操作结合使用
- **第3阶段**: 使用 SETP 和 SELP 进行条件原子操作
- **当前阶段**: 提供并发编程基础

完整的原子操作实现为 PTX VM 提供了基本的并发原语，是实现更复杂的多线程 CUDA 程序的重要基础。
