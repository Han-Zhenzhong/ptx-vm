# PTX 虚拟机 - 新功能实现指南

**Authors**: Han-Zhenzhong, TongyiLingma, GitHub Copilot  
**Last Updated**: 2025-10-29

## ✅ 已完成的改进

### 1. 参数传递修复（最高优先级） ✅

**文件**: `src/host/host_api.cpp`

**修改**: 在 `cuLaunchKernel` 中添加了参数复制到参数内存的逻辑

```cpp
// 将每个参数复制到参数内存
size_t offset = 0;
for (size_t i = 0; i < entryFunc.parameters.size(); ++i) {
    const PTXParameter& param = entryFunc.parameters[i];
    const uint8_t* paramData = static_cast<const uint8_t*>(kernelParams[i]);
    for (size_t j = 0; j < param.size; ++j) {
        mem.write<uint8_t>(MemorySpace::PARAMETER, 
                          0x1000 + offset + j, 
                          paramData[j]);
    }
    offset += param.size;
}
```

**影响**: 现在 `ld.param` 指令可以正确读取参数值！

---

### 2. 浮点寄存器支持 ✅

**文件**: `src/registers/register_bank.hpp`, `register_bank.cpp`

**新增API**:
```cpp
// 浮点寄存器操作
float readFloatRegister(size_t registerIndex) const;
void writeFloatRegister(size_t registerIndex, float value);
double readDoubleRegister(size_t registerIndex) const;
void writeDoubleRegister(size_t registerIndex, double value);
```

**实现细节**:
- 使用 `std::vector<uint64_t> m_floatRegisters` 存储
- 通过 `std::memcpy` 进行类型转换（避免类型双关问题）
- 支持 32 位和 64 位浮点数

---

### 3. 特殊寄存器支持 ✅

**文件**: `src/registers/register_bank.hpp`, `register_bank.cpp`

**新增API**:
```cpp
uint32_t readSpecialRegister(SpecialRegister reg) const;
void setThreadId(uint32_t x, uint32_t y, uint32_t z);
void setBlockId(uint32_t x, uint32_t y, uint32_t z);
void setThreadDimensions(uint32_t x, uint32_t y, uint32_t z);
void setGridDimensions(uint32_t x, uint32_t y, uint32_t z);
```

**支持的特殊寄存器**:
- `%tid.x/y/z` - 线程ID
- `%ctaid.x/y/z` - 块ID
- `%ntid.x/y/z` - 块大小
- `%nctaid.x/y/z` - 网格大小
- `%warpsize` - Warp大小（默认32）
- `%laneid` - Lane ID

---

### 4. 指令类型扩展 ✅

**文件**: `include/instruction_types.hpp`

**新增指令类型**:

```cpp
// 浮点指令
ADD_F32, ADD_F64, SUB_F32, SUB_F64,
MUL_F32, MUL_F64, DIV_F32, DIV_F64,
FMA_F32, FMA_F64, SQRT_F32, SQRT_F64,
// ... 等

// 比较和选择
SETP, SELP, SET,

// 类型转换
CVT,

// 原子操作
ATOM_ADD, ATOM_SUB, ATOM_EXCH, ATOM_CAS,
ATOM_MIN, ATOM_MAX, ATOM_AND, ATOM_OR, ATOM_XOR
```

**新增枚举**:
```cpp
enum class CompareOp { EQ, NE, LT, LE, GT, GE, ... };
enum class DataType { S8, S16, S32, S64, U8, ..., F32, F64 };
```

---

## 🚧 待完成的实现（代码示例）

### 5. 浮点指令实现

**步骤 1**: 在 `parser.cpp` 中添加浮点指令识别

```cpp
InstructionTypes PTXParser::Impl::opcodeToInstructionType(const std::string& opcode) {
    // 现有代码...
    
    // 🔧 新增：检查是否有 .f32 或 .f64 修饰符
    bool isF32 = (modifiers.find(".f32") != std::string::npos);
    bool isF64 = (modifiers.find(".f64") != std::string::npos);
    
    if (opcode == "add") {
        if (isF32) return InstructionTypes::ADD_F32;
        if (isF64) return InstructionTypes::ADD_F64;
        return InstructionTypes::ADD;  // 默认整数
    }
    
    if (opcode == "mul") {
        if (isF32) return InstructionTypes::MUL_F32;
        if (isF64) return InstructionTypes::MUL_F64;
        return InstructionTypes::MUL;
    }
    
    if (opcode == "fma") {
        if (isF32) return InstructionTypes::FMA_F32;
        if (isF64) return InstructionTypes::FMA_F64;
    }
    
    if (opcode == "sqrt") {
        if (isF32) return InstructionTypes::SQRT_F32;
        if (isF64) return InstructionTypes::SQRT_F64;
    }
    
    // ... 其他浮点指令
}
```

**步骤 2**: 在 `executor.cpp` 中添加执行函数

```cpp
// 文件：src/execution/executor.cpp

bool PTXExecutor::Impl::executeADD_F32(const DecodedInstruction& instr) {
    // 读取源操作数（浮点寄存器）
    float src1 = m_registerBank->readFloatRegister(instr.sources[0].registerIndex);
    float src2 = m_registerBank->readFloatRegister(instr.sources[1].registerIndex);
    
    // 执行浮点加法
    float result = src1 + src2;
    
    // 写回目标寄存器
    m_registerBank->writeFloatRegister(instr.dest.registerIndex, result);
    
    // 更新性能计数器
    m_performanceCounters->incrementCounter(PerformanceCounterIDs::INSTRUCTIONS_EXECUTED);
    
    return true;
}

bool PTXExecutor::Impl::executeMUL_F32(const DecodedInstruction& instr) {
    float src1 = m_registerBank->readFloatRegister(instr.sources[0].registerIndex);
    float src2 = m_registerBank->readFloatRegister(instr.sources[1].registerIndex);
    float result = src1 * src2;
    m_registerBank->writeFloatRegister(instr.dest.registerIndex, result);
    m_performanceCounters->incrementCounter(PerformanceCounterIDs::INSTRUCTIONS_EXECUTED);
    return true;
}

bool PTXExecutor::Impl::executeFMA_F32(const DecodedInstruction& instr) {
    // fma.f32 %f0, %f1, %f2, %f3;  // %f0 = %f1 * %f2 + %f3
    float src1 = m_registerBank->readFloatRegister(instr.sources[0].registerIndex);
    float src2 = m_registerBank->readFloatRegister(instr.sources[1].registerIndex);
    float src3 = m_registerBank->readFloatRegister(instr.sources[2].registerIndex);
    
    // 使用 FMA 指令（如果可用）或模拟
    float result = src1 * src2 + src3;
    
    m_registerBank->writeFloatRegister(instr.dest.registerIndex, result);
    m_performanceCounters->incrementCounter(PerformanceCounterIDs::INSTRUCTIONS_EXECUTED);
    return true;
}

bool PTXExecutor::Impl::executeSQRT_F32(const DecodedInstruction& instr) {
    float src = m_registerBank->readFloatRegister(instr.sources[0].registerIndex);
    float result = std::sqrt(src);
    m_registerBank->writeFloatRegister(instr.dest.registerIndex, result);
    m_performanceCounters->incrementCounter(PerformanceCounterIDs::INSTRUCTIONS_EXECUTED);
    return true;
}
```

**步骤 3**: 在 `executeDecodedInstruction` 中添加 case 分支

```cpp
bool PTXExecutor::Impl::executeDecodedInstruction(const DecodedInstruction& instr) {
    // ... 现有代码 ...
    
    switch (instr.type) {
        // ... 现有 case ...
        
        case InstructionTypes::ADD_F32:
            return executeADD_F32(instr);
        case InstructionTypes::MUL_F32:
            return executeMUL_F32(instr);
        case InstructionTypes::FMA_F32:
            return executeFMA_F32(instr);
        case InstructionTypes::SQRT_F32:
            return executeSQRT_F32(instr);
        
        // ... 其他浮点指令 ...
    }
}
```

---

### 6. SETP 指令实现

**步骤 1**: 解析 SETP 指令

```cpp
// parser.cpp
InstructionTypes PTXParser::Impl::opcodeToInstructionType(const std::string& opcode) {
    if (opcode == "setp") return InstructionTypes::SETP;
    // ...
}

// 解析比较操作符
CompareOp parseCompareOp(const std::vector<std::string>& modifiers) {
    for (const auto& mod : modifiers) {
        if (mod == ".lt") return CompareOp::LT;
        if (mod == ".le") return CompareOp::LE;
        if (mod == ".gt") return CompareOp::GT;
        if (mod == ".ge") return CompareOp::GE;
        if (mod == ".eq") return CompareOp::EQ;
        if (mod == ".ne") return CompareOp::NE;
    }
    return CompareOp::EQ;  // 默认
}
```

**步骤 2**: 执行 SETP 指令

```cpp
// executor.cpp
bool PTXExecutor::Impl::executeSETP(const DecodedInstruction& instr) {
    // setp.lt.s32 %p1, %r1, %r2;  // %p1 = (%r1 < %r2)
    
    bool result = false;
    
    // 根据数据类型读取源操作数
    if (instr.dataType == DataType::S32) {
        int32_t src1 = static_cast<int32_t>(
            m_registerBank->readRegister(instr.sources[0].registerIndex));
        int32_t src2 = static_cast<int32_t>(
            m_registerBank->readRegister(instr.sources[1].registerIndex));
        
        switch (instr.compareOp) {
            case CompareOp::LT: result = (src1 < src2); break;
            case CompareOp::LE: result = (src1 <= src2); break;
            case CompareOp::GT: result = (src1 > src2); break;
            case CompareOp::GE: result = (src1 >= src2); break;
            case CompareOp::EQ: result = (src1 == src2); break;
            case CompareOp::NE: result = (src1 != src2); break;
        }
    } else if (instr.dataType == DataType::F32) {
        float src1 = m_registerBank->readFloatRegister(instr.sources[0].registerIndex);
        float src2 = m_registerBank->readFloatRegister(instr.sources[1].registerIndex);
        
        switch (instr.compareOp) {
            case CompareOp::LT: result = (src1 < src2); break;
            case CompareOp::LE: result = (src1 <= src2); break;
            case CompareOp::GT: result = (src1 > src2); break;
            case CompareOp::GE: result = (src1 >= src2); break;
            case CompareOp::EQ: result = (src1 == src2); break;
            case CompareOp::NE: result = (src1 != src2); break;
        }
    }
    
    // 写入谓词寄存器
    m_registerBank->writePredicate(instr.dest.predicateIndex, result);
    
    m_performanceCounters->incrementCounter(PerformanceCounterIDs::INSTRUCTIONS_EXECUTED);
    return true;
}
```

---

### 7. SELP 指令实现

```cpp
bool PTXExecutor::Impl::executeSELP(const DecodedInstruction& instr) {
    // selp.s32 %r3, %r1, %r2, %p1;  // %r3 = %p1 ? %r1 : %r2
    
    // 读取谓词
    bool pred = m_registerBank->readPredicate(instr.sources[2].predicateIndex);
    
    // 根据数据类型读取源操作数
    if (instr.dataType == DataType::S32 || instr.dataType == DataType::U32) {
        uint64_t src1 = m_registerBank->readRegister(instr.sources[0].registerIndex);
        uint64_t src2 = m_registerBank->readRegister(instr.sources[1].registerIndex);
        uint64_t result = pred ? src1 : src2;
        m_registerBank->writeRegister(instr.dest.registerIndex, result);
    } else if (instr.dataType == DataType::F32) {
        float src1 = m_registerBank->readFloatRegister(instr.sources[0].registerIndex);
        float src2 = m_registerBank->readFloatRegister(instr.sources[1].registerIndex);
        float result = pred ? src1 : src2;
        m_registerBank->writeFloatRegister(instr.dest.registerIndex, result);
    }
    
    m_performanceCounters->incrementCounter(PerformanceCounterIDs::INSTRUCTIONS_EXECUTED);
    return true;
}
```

---

### 8. CVT 类型转换指令

```cpp
bool PTXExecutor::Impl::executeCVT(const DecodedInstruction& instr) {
    // cvt.s32.f32 %r1, %f1;  // %r1 = (int32_t)%f1
    
    if (instr.srcType == DataType::F32 && instr.dstType == DataType::S32) {
        float src = m_registerBank->readFloatRegister(instr.sources[0].registerIndex);
        int32_t dst = static_cast<int32_t>(src);
        m_registerBank->writeRegister(instr.dest.registerIndex, 
                                      static_cast<uint64_t>(dst));
    }
    else if (instr.srcType == DataType::S32 && instr.dstType == DataType::F32) {
        int32_t src = static_cast<int32_t>(
            m_registerBank->readRegister(instr.sources[0].registerIndex));
        float dst = static_cast<float>(src);
        m_registerBank->writeFloatRegister(instr.dest.registerIndex, dst);
    }
    else if (instr.srcType == DataType::F32 && instr.dstType == DataType::U32) {
        float src = m_registerBank->readFloatRegister(instr.sources[0].registerIndex);
        uint32_t dst = static_cast<uint32_t>(src);
        m_registerBank->writeRegister(instr.dest.registerIndex, 
                                      static_cast<uint64_t>(dst));
    }
    // ... 其他类型转换组合
    
    m_performanceCounters->incrementCounter(PerformanceCounterIDs::INSTRUCTIONS_EXECUTED);
    return true;
}
```

---

### 9. 原子操作指令

```cpp
bool PTXExecutor::Impl::executeATOM_ADD(const DecodedInstruction& instr) {
    // atom.global.add.u32 %r1, [%rd1], %r2;
    // %r1 = old value at [%rd1]
    // [%rd1] = [%rd1] + %r2
    
    uint64_t address = instr.sources[0].address;
    uint32_t addValue = static_cast<uint32_t>(
        m_registerBank->readRegister(instr.sources[1].registerIndex));
    
    // 🔒 原子操作：读取-修改-写入
    // 注意：这里需要加锁以确保原子性（多线程情况下）
    uint32_t oldValue = m_memorySubsystem->read<uint32_t>(
        instr.memorySpace, address);
    
    uint32_t newValue = oldValue + addValue;
    
    m_memorySubsystem->write<uint32_t>(
        instr.memorySpace, address, newValue);
    
    // 返回旧值
    if (instr.dest.type == OperandType::REGISTER) {
        m_registerBank->writeRegister(instr.dest.registerIndex, 
                                      static_cast<uint64_t>(oldValue));
    }
    
    m_performanceCounters->incrementCounter(PerformanceCounterIDs::INSTRUCTIONS_EXECUTED);
    return true;
}

bool PTXExecutor::Impl::executeATOM_CAS(const DecodedInstruction& instr) {
    // atom.global.cas.b32 %r1, [%rd1], %r2, %r3;
    // if ([%rd1] == %r2) [%rd1] = %r3;
    // %r1 = old value at [%rd1]
    
    uint64_t address = instr.sources[0].address;
    uint32_t compareValue = static_cast<uint32_t>(
        m_registerBank->readRegister(instr.sources[1].registerIndex));
    uint32_t newValue = static_cast<uint32_t>(
        m_registerBank->readRegister(instr.sources[2].registerIndex));
    
    uint32_t oldValue = m_memorySubsystem->read<uint32_t>(
        instr.memorySpace, address);
    
    if (oldValue == compareValue) {
        m_memorySubsystem->write<uint32_t>(
            instr.memorySpace, address, newValue);
    }
    
    // 返回旧值
    if (instr.dest.type == OperandType::REGISTER) {
        m_registerBank->writeRegister(instr.dest.registerIndex, 
                                      static_cast<uint64_t>(oldValue));
    }
    
    m_performanceCounters->incrementCounter(PerformanceCounterIDs::INSTRUCTIONS_EXECUTED);
    return true;
}
```

---

### 10. MOV 特殊寄存器支持

```cpp
bool PTXExecutor::Impl::executeMOV(const DecodedInstruction& instr) {
    // 现有代码...
    
    // 🔧 新增：支持从特殊寄存器移动
    if (instr.sources[0].type == OperandType::SPECIAL_REGISTER) {
        // mov.u32 %r1, %tid.x;
        SpecialRegister sreg = /* 从操作数解析 */;
        uint32_t value = m_registerBank->readSpecialRegister(sreg);
        m_registerBank->writeRegister(instr.dest.registerIndex, 
                                      static_cast<uint64_t>(value));
        return true;
    }
    
    // ... 现有的 MOV 逻辑
}
```

---

## 📝 完整实现清单

### 需要修改的文件

1. **include/instruction_types.hpp** ✅
   - 添加新指令类型枚举
   - 添加 CompareOp、DataType 枚举
   - 扩展 DecodedInstruction 结构

2. **src/registers/register_bank.hpp** ✅
   - 添加浮点寄存器API
   - 添加特殊寄存器API

3. **src/registers/register_bank.cpp** ✅
   - 实现浮点寄存器操作
   - 实现特殊寄存器操作

4. **src/parser/parser.cpp** 🚧
   - 修改 `opcodeToInstructionType` 识别新指令
   - 添加比较操作符解析
   - 添加数据类型解析

5. **src/execution/executor.cpp** 🚧
   - 添加所有新指令的执行函数
   - 在 `executeDecodedInstruction` 中添加 case 分支

6. **src/host/host_api.cpp** ✅
   - 修复参数传递（已完成）
   - 可选：在 cuLaunchKernel 中设置 grid/block 维度到特殊寄存器

---

## 🧪 测试用例示例

### 测试 1: 浮点运算

```ptx
.version 7.0
.target sm_50
.address_size 64

.entry test_float_add(
    .param .u64 result_ptr
)
{
    .reg .f32 %f<5>;
    .reg .u64 %rd<2>;
    
    // 加载参数
    ld.param.u64 %rd1, [result_ptr];
    
    // 浮点运算
    mov.f32 %f1, 3.14;
    mov.f32 %f2, 2.71;
    add.f32 %f3, %f1, %f2;  // %f3 = 5.85
    mul.f32 %f4, %f1, %f2;  // %f4 = 8.5094
    
    // 存储结果
    st.global.f32 [%rd1], %f3;
    st.global.f32 [%rd1+4], %f4;
    
    exit;
}
```

### 测试 2: 比较和分支

```ptx
.entry test_setp(
    .param .u64 data_ptr
)
{
    .reg .s32 %r<5>;
    .reg .pred %p<2>;
    .reg .u64 %rd1;
    
    ld.param.u64 %rd1, [data_ptr];
    
    mov.s32 %r1, 10;
    mov.s32 %r2, 20;
    
    setp.lt.s32 %p1, %r1, %r2;  // %p1 = true (10 < 20)
    selp.s32 %r3, %r1, %r2, %p1;  // %r3 = %r1 = 10
    
    st.global.s32 [%rd1], %r3;
    
    exit;
}
```

### 测试 3: 类型转换

```ptx
.entry test_cvt(
    .param .u64 result_ptr
)
{
    .reg .f32 %f<3>;
    .reg .s32 %r<3>;
    .reg .u64 %rd1;
    
    ld.param.u64 %rd1, [result_ptr];
    
    mov.f32 %f1, 3.14;
    cvt.s32.f32 %r1, %f1;  // %r1 = 3
    
    mov.s32 %r2, 42;
    cvt.f32.s32 %f2, %r2;  // %f2 = 42.0
    
    st.global.s32 [%rd1], %r1;
    st.global.f32 [%rd1+4], %f2;
    
    exit;
}
```

### 测试 4: 特殊寄存器

```ptx
.entry test_special_regs(
    .param .u64 result_ptr
)
{
    .reg .u32 %r<5>;
    .reg .u64 %rd1;
    
    ld.param.u64 %rd1, [result_ptr];
    
    mov.u32 %r1, %tid.x;    // 线程ID
    mov.u32 %r2, %ctaid.x;  // 块ID
    mov.u32 %r3, %ntid.x;   // 块大小
    mov.u32 %r4, %nctaid.x; // 网格大小
    
    st.global.u32 [%rd1], %r1;
    st.global.u32 [%rd1+4], %r2;
    st.global.u32 [%rd1+8], %r3;
    st.global.u32 [%rd1+12], %r4;
    
    exit;
}
```

### 测试 5: 原子操作

```ptx
.entry test_atomic(
    .param .u64 counter_ptr
)
{
    .reg .u32 %r<3>;
    .reg .u64 %rd1;
    
    ld.param.u64 %rd1, [counter_ptr];
    
    mov.u32 %r2, 1;
    atom.global.add.u32 %r1, [%rd1], %r2;  // 原子加1，返回旧值
    
    exit;
}
```

---

## 📊 实现优先级建议

### 第 1 优先级（立即实现）
1. ✅ 参数传递修复
2. ✅ 浮点寄存器支持
3. ✅ 特殊寄存器支持
4. 🚧 基本浮点指令（ADD_F32, MUL_F32, DIV_F32）
5. 🚧 SETP 指令

### 第 2 优先级（本周完成）
6. SELP 指令
7. CVT 基本类型转换
8. MOV 支持特殊寄存器

### 第 3 优先级（下周完成）
9. FMA, SQRT 等高级浮点指令
10. 原子操作基础（ATOM_ADD, ATOM_CAS）

---

## 🔗 相关文档

- `docs/comprehensive_implementation_analysis.md` - 完整分析报告
- `docs/multi_function_execution_guide.md` - 多函数执行指南
- `examples/simple_math_example.ptx` - 简单示例
- `examples/multi_function_example.ptx` - 多函数示例

---

## ⚠️ 注意事项

1. **寄存器索引**: 确保浮点寄存器和整数寄存器使用不同的索引空间
2. **类型安全**: 使用 `std::memcpy` 而非类型双关进行浮点/整数转换
3. **原子操作**: 在多线程环境下需要添加互斥锁
4. **特殊寄存器**: 需要在 cuLaunchKernel 中设置 grid/block 维度
5. **错误处理**: 添加详细的错误检查和日志输出

---

生成时间: 2025-10-26
