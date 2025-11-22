# PTX 解析器完整数据结构设计

**Authors**: Han-Zhenzhong, TongyiLingma, GitHub Copilot  
**Last Updated**: 2025-10-29

## 📋 概述

PTX 代码**可以包含多个函数**，包括：
- **`.entry`** - 内核入口点（可以有多个）
- **`.func`** - 设备函数（被内核或其他设备函数调用）

解析后的数据结构需要能够表示完整的程序结构，包括函数、参数、标签、变量等。

---

## 🏗️ 数据结构层次

```
PTXProgram (根容器)
├── PTXMetadata (文件元数据)
│   ├── version: "6.0"
│   ├── target: "sm_50"
│   ├── addressSize: 64
│   └── debugMode: false
│
├── instructions: vector<DecodedInstruction> (所有指令的线性序列)
│   ├── [0] mov.s32 %r1, 42
│   ├── [1] add.s32 %r2, %r1, %r1
│   └── ...
│
├── functions: vector<PTXFunction> (所有函数)
│   ├── [0] PTXFunction (add_numbers - device function)
│   │   ├── name: "add_numbers"
│   │   ├── isEntry: false
│   │   ├── parameters: [.param .s32 %a, .param .s32 %b]
│   │   ├── returnValues: [.reg .s32 %result]
│   │   ├── registerDeclarations: [.reg .s32 %r<4>]
│   │   ├── startInstructionIndex: 0
│   │   ├── endInstructionIndex: 5
│   │   └── localLabels: {}
│   │
│   ├── [1] PTXFunction (multiply_numbers)
│   │   └── ...
│   │
│   ├── [2] PTXFunction (kernel_process_array - kernel entry)
│   │   ├── name: "kernel_process_array"
│   │   ├── isEntry: true
│   │   ├── parameters: [input_ptr, output_ptr, size]
│   │   ├── returnValues: []
│   │   ├── startInstructionIndex: 20
│   │   ├── endInstructionIndex: 45
│   │   └── localLabels: {"loop_start": 25, "loop_end": 40}
│   │
│   └── [3] PTXFunction (kernel_simple_test)
│       └── ...
│
├── symbolTable: PTXSymbolTable (全局符号表)
│   ├── functions: map<string, PTXFunction>
│   │   ├── "add_numbers" → PTXFunction
│   │   ├── "multiply_numbers" → PTXFunction
│   │   ├── "kernel_process_array" → PTXFunction
│   │   └── "kernel_simple_test" → PTXFunction
│   │
│   ├── globalLabels: map<string, size_t>
│   │   └── (empty for this example)
│   │
│   ├── variables: map<string, PTXGlobalVariable>
│   │   └── (empty for this example)
│   │
│   └── parameterSymbols: map<string, PTXParameter*>
│       ├── "result_ptr" → PTXParameter*
│       ├── "a" → PTXParameter*
│       └── ...
│
├── globalVariables: vector<PTXGlobalVariable>
│   └── (empty for this example)
│
└── entryPoints: vector<size_t> (入口点索引)
    ├── [0] → 2 (kernel_process_array)
    └── [1] → 3 (kernel_simple_test)
```

---

## 📊 核心数据结构详解

### 1. PTXProgram - 根容器

这是**顶层数据结构**，包含解析后的所有信息：

```cpp
struct PTXProgram {
    PTXMetadata metadata;                    // 文件元数据
    std::vector<DecodedInstruction> instructions;  // 所有指令
    PTXSymbolTable symbolTable;              // 符号表
    std::vector<PTXFunction> functions;      // 所有函数
    std::vector<PTXGlobalVariable> globalVariables;
    std::vector<size_t> entryPoints;         // 入口点索引
    std::vector<std::string> errors;
    std::vector<std::string> warnings;
};
```

**使用方式**：
```cpp
PTXParser parser;
parser.parseFile("program.ptx");
const PTXProgram& program = parser.getProgram();
```

### 2. PTXFunction - 函数定义

表示一个 `.func` 或 `.entry`：

```cpp
struct PTXFunction {
    std::string name;                        // 函数名
    bool isEntry;                            // true=kernel, false=device function
    std::vector<PTXParameter> parameters;    // 参数
    std::vector<PTXParameter> returnValues;  // 返回值（仅 .func）
    std::vector<PTXRegisterDeclaration> registerDeclarations;
    size_t startInstructionIndex;            // 起始指令索引
    size_t endInstructionIndex;              // 结束指令索引
    std::map<std::string, size_t> localLabels; // 局部标签
};
```

**示例数据**（对应 `simple_math_example.ptx`）：

```cpp
PTXFunction {
    name: "simple_math_kernel",
    isEntry: true,
    parameters: [
        PTXParameter {
            name: "result_ptr",
            type: ".u64",
            offset: 0,
            size: 8,
            isPointer: true
        }
    ],
    returnValues: [],
    registerDeclarations: [
        PTXRegisterDeclaration {
            type: ".f32",
            baseRegister: "%f",
            startIndex: 0,
            count: 10  // %f0 到 %f9
        },
        PTXRegisterDeclaration {
            type: ".s32",
            baseRegister: "%r",
            startIndex: 0,
            count: 10  // %r0 到 %r9
        }
    ],
    startInstructionIndex: 0,
    endInstructionIndex: 12,
    localLabels: {}  // 没有标签
}
```

### 3. PTXParameter - 参数定义

```cpp
struct PTXParameter {
    std::string name;      // "result_ptr"
    std::string type;      // ".u64"
    size_t offset;         // 0 (在参数内存中的偏移)
    size_t size;           // 8 (字节)
    bool isPointer;        // true
};
```

### 4. PTXSymbolTable - 符号表

提供**快速查找**功能：

```cpp
struct PTXSymbolTable {
    std::map<std::string, PTXFunction> functions;
    std::map<std::string, size_t> globalLabels;
    std::map<std::string, PTXGlobalVariable> variables;
    std::map<std::string, PTXParameter*> parameterSymbols;
    
    // 查找函数
    const PTXFunction* findFunction(const std::string& name) const;
    
    // 查找标签（先局部再全局）
    bool findLabel(const std::string& name, 
                   const std::string& currentFunction, 
                   size_t& outIndex) const;
    
    // 查找参数
    const PTXParameter* findParameter(const std::string& name) const;
};
```

---

## 🔄 解析流程（两遍扫描）

### Pass 1: 收集元数据和符号

```
输入：PTX 源代码字符串

处理：
├── 解析 .version, .target, .address_size
├── 识别函数声明 (.entry, .func)
│   ├── 提取函数名
│   ├── 解析参数列表
│   └── 解析返回值
├── 记录标签位置
├── 解析寄存器声明 (.reg)
└── 解析全局变量 (.global, .shared, .const)

输出：
├── PTXMetadata
├── PTXSymbolTable (部分填充)
└── 函数边界信息
```

### Pass 2: 解析指令

```
输入：PTX 源代码 + Pass 1 的符号表

处理：
├── 逐行解析指令
├── 转换为 DecodedInstruction
├── 解析符号引用（使用符号表）
└── 分配指令到对应的函数

输出：
└── vector<DecodedInstruction>
```

---

## 💡 实际示例：multi_function_example.ptx

### 文件结构

```ptx
.version 6.0
.target sm_50

// Device function 1
.func (.reg .s32 %result) add_numbers (.param .s32 %a, .param .s32 %b)
{
    // 指令 0-5
}

// Device function 2
.func (.reg .s32 %product) multiply_numbers (.param .s32 %x, .param .s32 %y)
{
    // 指令 6-11
}

// Kernel entry 1
.entry kernel_process_array (.param .u64 input_ptr, ...)
{
    // 指令 12-40
}

// Kernel entry 2
.entry kernel_simple_test (.param .u64 data_ptr)
{
    // 指令 41-50
}
```

### 解析后的 PTXProgram

```cpp
PTXProgram {
    metadata: {
        version: "6.0",
        target: "sm_50",
        addressSize: 64
    },
    
    instructions: [
        // 0-5: add_numbers 的指令
        DecodedInstruction { type: LD_PARAM, ... },
        DecodedInstruction { type: LD_PARAM, ... },
        DecodedInstruction { type: ADD, ... },
        DecodedInstruction { type: ST_PARAM, ... },
        DecodedInstruction { type: RET, ... },
        
        // 6-11: multiply_numbers 的指令
        ...
        
        // 12-40: kernel_process_array 的指令
        ...
        
        // 41-50: kernel_simple_test 的指令
        ...
    ],
    
    functions: [
        PTXFunction {
            name: "add_numbers",
            isEntry: false,
            parameters: [
                {name: "a", type: ".s32", offset: 0, size: 4},
                {name: "b", type: ".s32", offset: 4, size: 4}
            ],
            returnValues: [
                {name: "result", type: ".s32", offset: 0, size: 4}
            ],
            startInstructionIndex: 0,
            endInstructionIndex: 4
        },
        
        PTXFunction {
            name: "multiply_numbers",
            isEntry: false,
            startInstructionIndex: 5,
            endInstructionIndex: 10
        },
        
        PTXFunction {
            name: "kernel_process_array",
            isEntry: true,
            parameters: [
                {name: "input_ptr", type: ".u64", offset: 0, size: 8},
                {name: "output_ptr", type: ".u64", offset: 8, size: 8},
                {name: "size", type: ".u32", offset: 16, size: 4}
            ],
            startInstructionIndex: 11,
            endInstructionIndex: 39
        },
        
        PTXFunction {
            name: "kernel_simple_test",
            isEntry: true,
            startInstructionIndex: 40,
            endInstructionIndex: 49
        }
    ],
    
    symbolTable: {
        functions: {
            "add_numbers" → functions[0],
            "multiply_numbers" → functions[1],
            "kernel_process_array" → functions[2],
            "kernel_simple_test" → functions[3]
        }
    },
    
    entryPoints: [2, 3]  // 索引到 functions 数组
}
```

---

## 🎯 使用示例

### 1. 启动指定的内核

```cpp
PTXParser parser;
parser.parseFile("multi_function_example.ptx");
const PTXProgram& program = parser.getProgram();

// 查找内核
const PTXFunction* kernel = program.getEntryByName("kernel_process_array");
if (kernel) {
    std::cout << "Launching kernel: " << kernel->name << std::endl;
    
    // 设置参数
    for (const auto& param : kernel->parameters) {
        std::cout << "  Param: " << param.name 
                  << " (" << param.type << ", offset=" << param.offset << ")" 
                  << std::endl;
    }
    
    // 执行指令范围
    for (size_t i = kernel->startInstructionIndex; 
         i <= kernel->endInstructionIndex; 
         ++i) {
        executeInstruction(program.instructions[i]);
    }
}
```

### 2. 解析符号引用

```cpp
// 在指令中遇到 "ld.param.u64 %r0, [result_ptr]"
std::string paramName = "result_ptr";

// 查找参数定义
const PTXParameter* param = program.symbolTable.findParameter(paramName);
if (param) {
    std::cout << "Parameter '" << paramName << "' found at offset " 
              << param->offset << std::endl;
    
    // 从参数内存读取
    uint64_t address = PARAM_MEMORY_BASE + param->offset;
    uint64_t value = memory.read<uint64_t>(address);
}
```

### 3. 处理函数调用

```cpp
// 遇到 "call (result), add_numbers, (a, b)"
std::string funcName = "add_numbers";

const PTXFunction* callee = program.symbolTable.findFunction(funcName);
if (callee) {
    // 保存当前执行状态
    saveCallContext();
    
    // 跳转到函数入口
    currentInstructionIndex = callee->startInstructionIndex;
    
    // 设置参数
    setupFunctionParameters(callee->parameters);
}
```

### 4. 处理标签跳转

```cpp
// 遇到 "bra loop_start"
std::string labelName = "loop_start";
std::string currentFunc = "kernel_process_array";

size_t targetIndex;
if (program.symbolTable.findLabel(labelName, currentFunc, targetIndex)) {
    // 跳转
    currentInstructionIndex = targetIndex;
} else {
    std::cerr << "Label '" << labelName << "' not found!" << std::endl;
}
```

---

## 📝 与当前实现的对比

| 特性 | 当前实现 | 完整设计 |
|------|---------|---------|
| 数据结构 | `vector<DecodedInstruction>` | `PTXProgram` |
| 函数支持 | ❌ 无 | ✅ 多函数支持 |
| 参数解析 | ❌ 跳过 | ✅ 完整解析 |
| 符号表 | ❌ 无 | ✅ 完整符号表 |
| 标签处理 | ❌ 跳过 | ✅ 局部+全局标签 |
| 寄存器声明 | ❌ 忽略 | ✅ 记录并验证 |
| 错误处理 | ❌ 基本 | ✅ 详细错误/警告 |

---

## 🚀 实现建议

### 阶段 1：扩展数据结构（1-2天）
- 在 `src/parser/parser.hpp` 中添加新结构
- 保持向后兼容（保留 `getInstructions()`）
- 添加 `getProgram()` 方法

### 阶段 2：实现两遍扫描（3-5天）
- Pass 1: 元数据和符号收集
- Pass 2: 指令解析和符号解析

### 阶段 3：集成到 VM（2-3天）
- 更新 `PTXVM` 使用新结构
- 实现函数调用支持
- 处理参数传递

### 阶段 4：测试和优化（2-3天）
- 多函数测试
- 性能优化
- 错误处理

---

## 总结

完整的 PTX 解析器数据结构应该是：

```
PTXProgram
  ├── 元数据（version, target, etc.）
  ├── 指令序列（线性存储）
  ├── 函数列表（支持多个 .func 和 .entry）
  ├── 符号表（快速查找）
  └── 入口点列表
```

这样的设计能够：
✅ 支持多函数 PTX 程序
✅ 正确处理参数传递
✅ 解析符号引用
✅ 支持函数调用
✅ 提供清晰的程序结构视图
