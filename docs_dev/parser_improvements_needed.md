# PTX 解析器改进需求分析

**Authors**: Han-Zhenzhong, TongyiLingma, GitHub Copilot  
**Last Updated**: 2025-10-29

## 📋 目录
- [当前状态](#当前状态)
- [主要缺失功能](#主要缺失功能)
- [详细改进建议](#详细改进建议)
- [实现优先级](#实现优先级)
- [参考资源](#参考资源)

---

## 当前状态

### ✅ 已实现的功能

当前解析器（`src/parser/parser.cpp`）已经实现：

1. **基础指令解析**：
   - 算术运算：`add`, `sub`, `mul`, `div`, `rem`
   - 逻辑运算：`and`, `or`, `xor`, `not`, `shl`, `shr`
   - 数据移动：`mov`, `ld`, `st`
   - 控制流：`bra`, `jump`, `call`, `ret`
   - 同步：`sync`, `membar`, `barrier`

2. **基础语法支持**：
   - 谓词执行：`@p0 add.s32 %r1, %r2, %r3`
   - 类型修饰符：`add.s32`, `ld.global.f32`
   - 注释过滤：`//` 和 `#`
   - 操作数解析：寄存器、立即数、内存地址

3. **错误处理**：
   - 文件打开失败检测
   - 基本的语法错误处理

---

## 主要缺失功能

### ❌ 1. 缺失的指令类型

#### 比较和选择指令（Critical）
```ptx
// 比较设置谓词 - 示例文件中大量使用
setp.eq.s32 %p0, %r1, %r2;       // ❌ 未支持
setp.lt.s32 %p1, %r4, %r5;       // ❌ 未支持
setp.gt.f32 %p2, %f1, %f2;       // ❌ 未支持
setp.le.u64 %p3, %r1, 100;       // ❌ 未支持

// 条件选择
selp.s32 %r1, %r2, %r3, %p0;     // ❌ 未支持
```

#### 类型转换指令（High Priority）
```ptx
cvt.s32.f32 %r1, %f1;            // ❌ 未支持 - float to int
cvt.f32.s32 %f1, %r1;            // ❌ 未支持 - int to float
cvt.u64.u32 %rd1, %r1;           // ❌ 未支持 - 32-bit to 64-bit
cvt.rn.f32.f64 %f1, %fd1;        // ❌ 未支持 - double to float with rounding
```

#### 融合乘加指令（High Priority）
```ptx
mad.lo.s32 %r1, %r2, %r3, %r4;   // ❌ 未支持 - r1 = r2 * r3 + r4
mad.wide.s32 %rd1, %r1, %r2, %rd2; // ❌ 未支持
fma.rn.f32 %f1, %f2, %f3, %f4;   // ❌ 未支持 - IEEE 754 FMA
```

#### 最小/最大值指令（Medium Priority）
```ptx
min.s32 %r1, %r2, %r3;           // ❌ 未支持
max.f32 %f1, %f2, %f3;           // ❌ 未支持
```

#### 位操作指令（Medium Priority）
```ptx
clz.b32 %r1, %r2;                // ❌ 未支持 - count leading zeros
popc.b32 %r1, %r2;               // ❌ 未支持 - population count
brev.b32 %r1, %r2;               // ❌ 未支持 - bit reverse
bfe.u32 %r1, %r2, 8, 4;          // ❌ 未支持 - bit field extract
bfi.b32 %r1, %r2, %r3, 8, 4;     // ❌ 未支持 - bit field insert
```

#### 原子操作指令（High Priority）
```ptx
atom.global.add.u32 %r1, [%rd1], %r2;     // ❌ 未支持
atom.shared.cas.b32 %r1, [%r2], %r3, %r4; // ❌ 未支持
atom.global.exch.b32 %r1, [%rd1], %r2;    // ❌ 未支持
atom.global.min.u32 %r1, [%rd1], %r2;     // ❌ 未支持
atom.global.max.s32 %r1, [%rd1], %r2;     // ❌ 未支持
```

#### 特殊函数指令（Low Priority）
```ptx
sin.approx.f32 %f1, %f2;         // ❌ 未支持
cos.approx.f32 %f1, %f2;         // ❌ 未支持
sqrt.approx.f32 %f1, %f2;        // ❌ 未支持
rsqrt.approx.f32 %f1, %f2;       // ❌ 未支持
ex2.approx.f32 %f1, %f2;         // ❌ 未支持 - 2^x
lg2.approx.f32 %f1, %f2;         // ❌ 未支持 - log2(x)
```

#### 投票和归约指令（Medium Priority）
```ptx
vote.all.pred %p1, %p0;          // ❌ 未支持
vote.any.pred %p1, %p0;          // ❌ 未支持
vote.uni.pred %p1, %p0;          // ❌ 未支持
vote.ballot.b32 %r1, %p0;        // ❌ 未支持
```

#### 纹理和表面指令（Low Priority）
```ptx
tex.2d.v4.f32.f32 {%f1,%f2,%f3,%f4}, [tex, {%f5,%f6}];  // ❌ 未支持
suld.b.2d.v4.b32 {%r1,%r2,%r3,%r4}, [surf, {%r5,%r6}]; // ❌ 未支持
sust.b.2d.v4.b32 [surf, {%r1,%r2}], {%r3,%r4,%r5,%r6}; // ❌ 未支持
```

---

### ❌ 2. 缺失的元数据解析

当前解析器**完全跳过**了这些重要的 PTX 元数据：

#### PTX 版本和目标信息
```ptx
.version 6.0          // ❌ 未解析 - 应该记录 PTX 版本
.target sm_50         // ❌ 未解析 - 应该记录目标架构
.target sm_75, debug  // ❌ 未解析 - 可能有调试选项
.address_size 64      // ❌ 未解析 - 应该设置地址大小
```

**影响**：无法验证指令兼容性，无法根据架构优化。

#### 函数和内核声明
```ptx
.entry my_kernel (              // ❌ 未解析 - 应该标记入口点
    .param .u64 input_ptr,      // ❌ 未解析 - 应该创建参数列表
    .param .u64 output_ptr,
    .param .u32 size
)

.func (.reg .s32 %ret) my_func (  // ❌ 未解析 - 应该创建函数符号
    .param .s32 %a,
    .param .s32 %b
)
```

**影响**：无法正确处理函数调用，无法设置内核参数。

#### 寄存器声明
```ptx
.reg .f32 %f<10>;     // ❌ 未解析 - 应该预分配 f0-f9
.reg .s32 %r<20>;     // ❌ 未解析 - 应该预分配 r0-r19
.reg .pred %p<5>;     // ❌ 未解析 - 应该预分配 p0-p4
.reg .b64 %rd<8>;     // ❌ 未解析 - 64位寄存器
```

**影响**：无法进行寄存器分配检查，可能导致寄存器冲突。

#### 共享内存和常量内存声明
```ptx
.shared .align 4 .b8 shared_mem[4096];  // ❌ 未解析
.const .align 8 .f32 const_data[256];   // ❌ 未解析
.global .align 16 .v4 .f32 global_array[1024]; // ❌ 未解析
```

**影响**：无法分配共享内存，无法访问常量内存。

---

### ❌ 3. 标签和符号处理不完整

#### 标签定义
```ptx
loop_start:                    // ❌ 被跳过，没有记录地址映射
    setp.lt.s32 %p1, %r4, %r5;
    @%p1 bra loop_start;       // ❌ 无法解析符号跳转

BB0_1:                         // ❌ 基本块标签未处理
    add.s32 %r1, %r1, 1;
    bra.uni BB0_2;
```

**当前行为**：标签行被 `parsePTXInstruction` 直接返回 `false` 跳过。

**应该做的**：
1. 记录标签名 → 指令地址的映射
2. 在第二遍扫描时解析符号引用
3. 支持前向引用

#### 符号引用
```ptx
ld.global.u64 %rd1, [my_global_var];   // ❌ 符号 my_global_var 无法解析
call my_device_function, (%r1, %r2);   // ❌ 函数符号无法解析
```

---

### ❌ 4. 高级语法特性缺失

#### 向量操作
```ptx
ld.global.v4.f32 {%f0,%f1,%f2,%f3}, [%rd1];  // ❌ 向量加载未支持
st.global.v2.u32 [%rd1], {%r1,%r2};          // ❌ 向量存储未支持
```

#### 地址偏移计算
```ptx
ld.global.s32 %r1, [%rd1+16];          // ✅ 可能支持（取决于 parseOperand）
ld.global.s32 %r1, [%rd1+%r2];         // ❌ 寄存器偏移未支持
ld.global.s32 %r1, [%rd1+%r2*4];       // ❌ 缩放偏移未支持
```

#### 特殊寄存器
```ptx
mov.u32 %r1, %tid.x;          // ❌ 线程ID寄存器未支持
mov.u32 %r2, %ctaid.x;        // ❌ CTA ID寄存器未支持
mov.u32 %r3, %ntid.x;         // ❌ 块大小寄存器未支持
mov.u32 %r4, %nctaid.x;       // ❌ 网格大小寄存器未支持
mov.u32 %r5, %warpid;         // ❌ Warp ID 未支持
mov.u32 %r6, %laneid;         // ❌ Lane ID 未支持
mov.u64 %rd1, %clock64;       // ❌ 时钟寄存器未支持
```

#### 指令修饰符
```ptx
add.cc.s32 %r1, %r2, %r3;     // ❌ .cc (carry out) 未处理
addc.s32 %r4, %r5, %r6;       // ❌ addc (add with carry) 未支持
div.approx.f32 %f1, %f2, %f3; // ❌ .approx 修饰符未处理
ld.global.ca.f32 %f1, [%rd1]; // ❌ .ca (cache all) 未处理
st.global.wb.s32 [%rd1], %r1; // ❌ .wb (write back) 未处理
```

---

## 详细改进建议

### 🔧 Phase 1: 扩展指令类型枚举

**文件**: `include/instruction_types.hpp`

```cpp
enum class InstructionTypes {
    // ... 现有指令 ...
    
    // 比较指令（Critical）
    SETP_EQ,    // Set predicate if equal
    SETP_NE,    // Set predicate if not equal
    SETP_LT,    // Set predicate if less than
    SETP_LE,    // Set predicate if less than or equal
    SETP_GT,    // Set predicate if greater than
    SETP_GE,    // Set predicate if greater than or equal
    
    // 选择指令
    SELP,       // Select based on predicate
    
    // 类型转换
    CVT,        // Convert type
    
    // 融合乘加
    MAD,        // Multiply-add
    FMA,        // Fused multiply-add
    
    // 最小最大
    MIN,
    MAX,
    
    // 位操作
    CLZ,        // Count leading zeros
    POPC,       // Population count
    BREV,       // Bit reverse
    BFE,        // Bit field extract
    BFI,        // Bit field insert
    
    // 原子操作
    ATOM_ADD,
    ATOM_SUB,
    ATOM_MIN,
    ATOM_MAX,
    ATOM_INC,
    ATOM_DEC,
    ATOM_CAS,   // Compare and swap
    ATOM_EXCH,  // Exchange
    
    // 特殊函数
    SIN,
    COS,
    SQRT,
    RSQRT,
    EX2,        // 2^x
    LG2,        // log2
    
    // 投票指令
    VOTE_ALL,
    VOTE_ANY,
    VOTE_UNI,
    VOTE_BALLOT,
    
    // 纹理和表面（低优先级）
    TEX,
    SULD,
    SUST,
    
    // 特殊寄存器移动
    MOV_SPECIAL,
    
    // 退出
    EXIT,
    
    MAX_INSTRUCTION_TYPE
};
```

### 🔧 Phase 2: 增强 Parser 数据结构

**文件**: `src/parser/parser.hpp`

```cpp
// PTX 元数据结构
struct PTXMetadata {
    std::string version;           // .version 6.0
    std::string target;            // .target sm_50
    int addressSize;               // .address_size 64
    bool debugMode;                // .target sm_50, debug
};

// PTX 参数定义
struct PTXParameter {
    std::string name;              // 参数名
    std::string type;              // .u64, .f32, etc.
    size_t offset;                 // 在参数内存中的偏移
    size_t size;                   // 大小（字节）
};

// PTX 函数/内核定义
struct PTXFunction {
    std::string name;              // 函数名
    bool isEntry;                  // .entry 还是 .func
    std::vector<PTXParameter> params;  // 参数列表
    std::vector<PTXParameter> returnValues; // 返回值（仅 .func）
    size_t startAddress;           // 第一条指令的地址
    size_t endAddress;             // 最后一条指令的地址
};

// PTX 符号表
struct PTXSymbolTable {
    std::map<std::string, size_t> labels;        // 标签名 → 指令地址
    std::map<std::string, PTXFunction> functions; // 函数名 → 函数定义
    std::map<std::string, uint64_t> variables;   // 变量名 → 内存地址
};

class PTXParser {
public:
    // ... 现有方法 ...
    
    // 新增：获取元数据
    const PTXMetadata& getMetadata() const;
    
    // 新增：获取符号表
    const PTXSymbolTable& getSymbolTable() const;
    
    // 新增：获取函数列表
    const std::vector<PTXFunction>& getFunctions() const;
    
    // 新增：通过名称查找函数
    const PTXFunction* findFunction(const std::string& name) const;
    
private:
    class Impl;
    std::unique_ptr<Impl> pImpl;
};
```

### 🔧 Phase 3: 实现多遍扫描解析

**文件**: `src/parser/parser.cpp`

```cpp
class PTXParser::Impl {
public:
    bool parseString(const std::string& ptxCode) {
        m_instructions.clear();
        m_errorMessage = "";
        m_metadata = PTXMetadata{};
        m_symbolTable = PTXSymbolTable{};
        
        // Pass 1: 解析元数据、函数声明、标签
        if (!firstPass(ptxCode)) {
            return false;
        }
        
        // Pass 2: 解析指令，解析符号引用
        if (!secondPass(ptxCode)) {
            return false;
        }
        
        return true;
    }
    
private:
    // 第一遍：收集元数据和符号
    bool firstPass(const std::string& ptxCode) {
        std::istringstream iss(ptxCode);
        std::string line;
        size_t instructionIndex = 0;
        PTXFunction* currentFunction = nullptr;
        
        while (std::getline(iss, line)) {
            line = trim(line);
            if (line.empty() || line[0] == '/' && line[1] == '/') {
                continue;
            }
            
            // 解析 .version
            if (line.find(".version") == 0) {
                m_metadata.version = extractValue(line, ".version");
                continue;
            }
            
            // 解析 .target
            if (line.find(".target") == 0) {
                m_metadata.target = extractValue(line, ".target");
                m_metadata.debugMode = (line.find("debug") != std::string::npos);
                continue;
            }
            
            // 解析 .address_size
            if (line.find(".address_size") == 0) {
                std::string sizeStr = extractValue(line, ".address_size");
                m_metadata.addressSize = std::stoi(sizeStr);
                continue;
            }
            
            // 解析 .entry 或 .func
            if (line.find(".entry") == 0 || line.find(".func") == 0) {
                currentFunction = parseFunction(line, iss);
                if (currentFunction) {
                    currentFunction->startAddress = instructionIndex;
                }
                continue;
            }
            
            // 解析标签
            if (line.back() == ':' && line.find('.') != 0) {
                std::string labelName = line.substr(0, line.size() - 1);
                m_symbolTable.labels[labelName] = instructionIndex;
                continue;
            }
            
            // 跳过其他非指令行
            if (line[0] == '.' || line[0] == '{' || line[0] == '}') {
                continue;
            }
            
            // 如果是指令，增加计数
            instructionIndex++;
        }
        
        return true;
    }
    
    // 第二遍：解析指令
    bool secondPass(const std::string& ptxCode) {
        std::istringstream iss(ptxCode);
        std::string line;
        
        while (std::getline(iss, line)) {
            line = trim(line);
            if (line.empty() || line[0] == '/' && line[1] == '/') {
                continue;
            }
            
            // 跳过元数据和标签
            if (line[0] == '.' || line.back() == ':') {
                continue;
            }
            
            // 解析指令
            PTXInstruction ptInstruction;
            if (parsePTXInstruction(line, ptInstruction)) {
                DecodedInstruction instruction = convertToDecoded(ptInstruction);
                m_instructions.push_back(instruction);
            }
        }
        
        return true;
    }
    
    // 解析函数定义
    PTXFunction* parseFunction(const std::string& line, std::istringstream& stream) {
        PTXFunction func;
        func.isEntry = (line.find(".entry") == 0);
        
        // 提取函数名
        // 例如：.entry my_kernel (
        // 或：   .func (.reg .s32 %ret) my_func (
        std::regex funcRegex(R"(\.(?:entry|func)\s+(?:\([^)]+\)\s+)?(\w+)\s*\()");
        std::smatch matches;
        if (std::regex_search(line, matches, funcRegex)) {
            func.name = matches[1].str();
        }
        
        // 解析参数（可能跨多行）
        std::string paramBlock = line;
        while (paramBlock.find(')') == std::string::npos) {
            std::string nextLine;
            if (!std::getline(stream, nextLine)) break;
            paramBlock += " " + trim(nextLine);
        }
        
        // 提取参数列表
        func.params = parseParameters(paramBlock);
        
        // 添加到符号表
        m_symbolTable.functions[func.name] = func;
        
        return &m_symbolTable.functions[func.name];
    }
    
    // 解析参数
    std::vector<PTXParameter> parseParameters(const std::string& paramBlock) {
        std::vector<PTXParameter> params;
        // TODO: 实现参数解析
        // 例如：.param .u64 input_ptr, .param .u32 size
        return params;
    }
    
    PTXMetadata m_metadata;
    PTXSymbolTable m_symbolTable;
    std::vector<PTXFunction> m_functions;
    // ... 其他成员 ...
};
```

### 🔧 Phase 4: 扩展操作码映射

**文件**: `src/parser/parser.cpp`

```cpp
InstructionTypes opcodeFromString(const std::string& op) {
    static const std::unordered_map<std::string, InstructionTypes> table = {
        // 现有指令...
        {"add", InstructionTypes::ADD},
        {"sub", InstructionTypes::SUB},
        // ... 
        
        // 新增：比较指令
        {"setp.eq", InstructionTypes::SETP_EQ},
        {"setp.ne", InstructionTypes::SETP_NE},
        {"setp.lt", InstructionTypes::SETP_LT},
        {"setp.le", InstructionTypes::SETP_LE},
        {"setp.gt", InstructionTypes::SETP_GT},
        {"setp.ge", InstructionTypes::SETP_GE},
        
        // 新增：选择指令
        {"selp", InstructionTypes::SELP},
        
        // 新增：类型转换
        {"cvt", InstructionTypes::CVT},
        
        // 新增：融合乘加
        {"mad", InstructionTypes::MAD},
        {"mad.lo", InstructionTypes::MAD},
        {"mad.hi", InstructionTypes::MAD},
        {"mad.wide", InstructionTypes::MAD},
        {"fma", InstructionTypes::FMA},
        
        // 新增：最小最大
        {"min", InstructionTypes::MIN},
        {"max", InstructionTypes::MAX},
        
        // 新增：位操作
        {"clz", InstructionTypes::CLZ},
        {"popc", InstructionTypes::POPC},
        {"brev", InstructionTypes::BREV},
        {"bfe", InstructionTypes::BFE},
        {"bfi", InstructionTypes::BFI},
        
        // 新增：原子操作
        {"atom.add", InstructionTypes::ATOM_ADD},
        {"atom.global.add", InstructionTypes::ATOM_ADD},
        {"atom.shared.add", InstructionTypes::ATOM_ADD},
        {"atom.sub", InstructionTypes::ATOM_SUB},
        {"atom.min", InstructionTypes::ATOM_MIN},
        {"atom.max", InstructionTypes::ATOM_MAX},
        {"atom.inc", InstructionTypes::ATOM_INC},
        {"atom.dec", InstructionTypes::ATOM_DEC},
        {"atom.cas", InstructionTypes::ATOM_CAS},
        {"atom.exch", InstructionTypes::ATOM_EXCH},
        
        // 新增：特殊函数
        {"sin", InstructionTypes::SIN},
        {"cos", InstructionTypes::COS},
        {"sqrt", InstructionTypes::SQRT},
        {"rsqrt", InstructionTypes::RSQRT},
        {"ex2", InstructionTypes::EX2},
        {"lg2", InstructionTypes::LG2},
        
        // 新增：投票指令
        {"vote.all", InstructionTypes::VOTE_ALL},
        {"vote.any", InstructionTypes::VOTE_ANY},
        {"vote.uni", InstructionTypes::VOTE_UNI},
        {"vote.ballot", InstructionTypes::VOTE_BALLOT},
        
        // 新增：退出
        {"exit", InstructionTypes::EXIT},
    };
    
    auto it = table.find(op);
    if (it != table.end()) return it->second;
    
    // 后备匹配：原子操作（支持 atom.<space>.<op> 格式）
    if (op.find("atom.") == 0) {
        if (op.find(".add") != std::string::npos) return InstructionTypes::ATOM_ADD;
        if (op.find(".sub") != std::string::npos) return InstructionTypes::ATOM_SUB;
        if (op.find(".cas") != std::string::npos) return InstructionTypes::ATOM_CAS;
        if (op.find(".exch") != std::string::npos) return InstructionTypes::ATOM_EXCH;
        // ... 其他原子操作
    }
    
    // 后备匹配：ld.*/st.*
    if (op.find("ld.") == 0) return InstructionTypes::LD;
    if (op.find("st.") == 0) return InstructionTypes::ST;
    
    return InstructionTypes::MAX_INSTRUCTION_TYPE;
}
```

### 🔧 Phase 5: 处理特殊寄存器

**文件**: `src/parser/parser.cpp`

```cpp
Operand parseOperand(const std::string& s) {
    Operand operand = {};
    std::string str = s;
    
    // 检查是否是特殊寄存器
    if (isSpecialRegister(str)) {
        operand.type = OperandType::SPECIAL_REGISTER;
        operand.specialRegType = getSpecialRegisterType(str);
        return operand;
    }
    
    // 现有逻辑...
    if (!str.empty() && str.front() == '[' && str.back() == ']') {
        operand.type = OperandType::MEMORY;
        // ... 解析内存地址表达式
        parseMemoryAddress(str, operand);
    } else if (!str.empty() && str[0] == '%') {
        operand.type = OperandType::REGISTER;
        // ...
    }
    // ...
    
    return operand;
}

bool isSpecialRegister(const std::string& str) {
    return str == "%tid.x" || str == "%tid.y" || str == "%tid.z" ||
           str == "%ctaid.x" || str == "%ctaid.y" || str == "%ctaid.z" ||
           str == "%ntid.x" || str == "%ntid.y" || str == "%ntid.z" ||
           str == "%nctaid.x" || str == "%nctaid.y" || str == "%nctaid.z" ||
           str == "%warpid" || str == "%laneid" || str == "%clock64";
}

// 解析内存地址表达式（支持偏移和索引）
void parseMemoryAddress(const std::string& addrExpr, Operand& operand) {
    // 去除 []
    std::string inner = addrExpr.substr(1, addrExpr.size() - 2);
    
    // 检查是否有偏移：[%rd1+16] 或 [%rd1+%r2]
    size_t plusPos = inner.find('+');
    if (plusPos != std::string::npos) {
        std::string base = trim(inner.substr(0, plusPos));
        std::string offset = trim(inner.substr(plusPos + 1));
        
        operand.baseRegister = parseRegisterIndex(base);
        
        if (offset.find('%') == 0) {
            // 寄存器偏移
            operand.hasRegisterOffset = true;
            operand.offsetRegister = parseRegisterIndex(offset);
        } else if (offset.find('*') != std::string::npos) {
            // 缩放偏移：%r2*4
            // TODO: 解析缩放因子
        } else {
            // 立即数偏移
            operand.hasImmediateOffset = true;
            operand.immediateOffset = std::stoll(offset);
        }
    } else {
        // 简单地址
        if (inner[0] == '%') {
            operand.baseRegister = parseRegisterIndex(inner);
        } else {
            // 符号地址
            operand.symbolName = inner;
        }
    }
}
```

---

## 实现优先级

### 🔴 **Critical（立即实施）**

1. **setp 系列指令**
   - 原因：示例代码中大量使用，当前无法运行任何控制流程序
   - 工作量：小（1-2 天）
   - 文件：`instruction_types.hpp`, `parser.cpp`, `executor.cpp`

2. **标签和符号表**
   - 原因：分支指令无法正确跳转
   - 工作量：中（3-5 天）
   - 文件：`parser.hpp`, `parser.cpp`

3. **exit 指令**
   - 原因：内核无法正确结束
   - 工作量：极小（1 小时）

### 🟠 **High Priority（尽快实施）**

4. **cvt 类型转换指令**
   - 原因：混合精度计算常见
   - 工作量：中（2-3 天）

5. **mad/fma 融合乘加**
   - 原因：高性能计算核心指令
   - 工作量：小（1-2 天）

6. **原子操作**
   - 原因：并发编程必需
   - 工作量：中（3-5 天）

7. **元数据解析（.version, .target, .entry, .param）**
   - 原因：正确设置执行环境
   - 工作量：中（3-5 天）

### 🟡 **Medium Priority（渐进实施）**

8. **selp, min, max 指令**
   - 工作量：小（1-2 天）

9. **位操作指令（clz, popc, brev）**
   - 工作量：小（1-2 天）

10. **特殊寄存器（%tid, %ctaid, etc.）**
    - 工作量：中（2-3 天）

11. **投票指令（vote.*）**
    - 工作量：中（2-3 天）

### 🟢 **Low Priority（可延后）**

12. **特殊函数（sin, cos, sqrt）**
    - 原因：可用标准库替代
    - 工作量：小

13. **纹理和表面指令**
    - 原因：高级特性，使用较少
    - 工作量：大

---

## 测试建议

为每个新增的指令类型添加测试：

```cpp
// tests/parser_tests/test_setp.cpp
TEST(ParserTest, ParseSetpInstruction) {
    PTXParser parser;
    std::string code = "setp.lt.s32 %p1, %r4, %r5;";
    ASSERT_TRUE(parser.parseString(code));
    
    const auto& instr = parser.getInstructions()[0];
    EXPECT_EQ(instr.type, InstructionTypes::SETP_LT);
    EXPECT_TRUE(instr.hasPredicate);
}

// tests/parser_tests/test_labels.cpp
TEST(ParserTest, ParseLabels) {
    PTXParser parser;
    std::string code = R"(
        mov.s32 %r1, 0;
    loop_start:
        add.s32 %r1, %r1, 1;
        bra loop_start;
    )";
    ASSERT_TRUE(parser.parseString(code));
    
    const auto& symbols = parser.getSymbolTable();
    ASSERT_TRUE(symbols.labels.count("loop_start") > 0);
    EXPECT_EQ(symbols.labels.at("loop_start"), 1);  // 第二条指令
}
```

---

## 参考资源

1. **NVIDIA PTX ISA 规范**
   - 官方文档：https://docs.nvidia.com/cuda/parallel-thread-execution/
   - 版本：7.0 及以上

2. **PTX 指令集参考**
   - 第 9 章：指令集（Instructions）
   - 附录 A：PTX 汇编器指令

3. **现有开源项目**
   - GPGPU-Sim：完整的 GPU 模拟器，包含 PTX 解析器
   - NVIDIA cuobjdump：PTX 反汇编工具

4. **测试用例来源**
   - CUDA SDK 示例
   - NVIDIA PTX 单元测试
   - 使用 `nvcc -ptx` 生成的真实 PTX 代码

---

## 总结

当前 PTX 解析器是一个**功能原型（Prototype）**，覆盖了基本的算术和内存操作，但缺失了：

1. ❌ **50%+ 的常用指令**（setp, cvt, mad, atom, etc.）
2. ❌ **所有元数据解析**（.version, .target, .entry, .reg, etc.）
3. ❌ **符号和标签处理**（无法正确处理控制流）
4. ❌ **高级语法特性**（向量操作、特殊寄存器、复杂寻址）

**建议的开发路线**：
- **Week 1-2**: 实现 Critical 优先级项（setp, 标签, exit）
- **Week 3-5**: 实现 High Priority 项（cvt, mad, atom, 元数据）
- **Week 6+**: 渐进实现 Medium 和 Low Priority 项

完成这些改进后，PTX-VM 将能够运行**真实的 CUDA 程序**生成的 PTX 代码，而不仅仅是简化的测试用例。
