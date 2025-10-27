// PTX Parser Complete Data Structures Design
// 完整的 PTX 解析结果数据结构设计

#ifndef PTX_PARSER_STRUCTURES_HPP
#define PTX_PARSER_STRUCTURES_HPP

#include <string>
#include <vector>
#include <map>
#include <unordered_map>
#include <cstdint>
#include "instruction_types.hpp"

// ========================================
// 1. PTX 元数据（文件级别）
// ========================================

struct PTXMetadata {
    std::string version;           // .version 6.0
    std::string target;            // .target sm_50
    int addressSize;               // .address_size 64
    bool debugMode;                // .target sm_50, debug
    
    PTXMetadata() : version(""), target(""), addressSize(64), debugMode(false) {}
};

// ========================================
// 2. 参数定义
// ========================================

struct PTXParameter {
    std::string name;              // 参数名称，如 "result_ptr", "a", "b"
    std::string type;              // 参数类型，如 ".u64", ".s32", ".f32"
    size_t offset;                 // 在参数内存空间中的偏移量
    size_t size;                   // 参数大小（字节）
    bool isPointer;                // 是否是指针类型（.u64, .s64）
    
    PTXParameter() : name(""), type(""), offset(0), size(0), isPointer(false) {}
    
    PTXParameter(const std::string& n, const std::string& t, size_t off, size_t sz, bool ptr = false)
        : name(n), type(t), offset(off), size(sz), isPointer(ptr) {}
};

// ========================================
// 3. 寄存器声明
// ========================================

struct PTXRegisterDeclaration {
    std::string type;              // 寄存器类型：.s32, .u64, .f32, .pred, etc.
    std::string baseRegister;      // 基础寄存器名：%r, %f, %p, %rd
    int startIndex;                // 起始索引
    int count;                     // 数量
    
    PTXRegisterDeclaration() : type(""), baseRegister(""), startIndex(0), count(0) {}
    
    // 例如：.reg .s32 %r<10> 表示 %r0 到 %r9
    PTXRegisterDeclaration(const std::string& t, const std::string& base, int start, int cnt)
        : type(t), baseRegister(base), startIndex(start), count(cnt) {}
};

// ========================================
// 4. 标签定义
// ========================================

struct PTXLabel {
    std::string name;              // 标签名称，如 "loop_start", "BB0_1"
    size_t instructionIndex;       // 对应的指令索引
    
    PTXLabel() : name(""), instructionIndex(0) {}
    
    PTXLabel(const std::string& n, size_t idx) : name(n), instructionIndex(idx) {}
};

// ========================================
// 5. 函数定义（包括 .func 和 .entry）
// ========================================

struct PTXFunction {
    std::string name;                          // 函数名称
    bool isEntry;                              // true = .entry (kernel), false = .func (device function)
    
    // 参数
    std::vector<PTXParameter> parameters;      // 输入参数列表
    std::vector<PTXParameter> returnValues;    // 返回值列表（仅 .func）
    
    // 寄存器声明
    std::vector<PTXRegisterDeclaration> registerDeclarations;
    
    // 代码范围
    size_t startInstructionIndex;              // 函数体第一条指令的索引
    size_t endInstructionIndex;                // 函数体最后一条指令的索引
    
    // 局部符号表（函数内的标签）
    std::map<std::string, size_t> localLabels; // 标签名 → 指令索引
    
    PTXFunction() : name(""), isEntry(false), startInstructionIndex(0), endInstructionIndex(0) {}
};

// ========================================
// 6. 全局变量/常量定义
// ========================================

struct PTXGlobalVariable {
    std::string name;              // 变量名称
    std::string type;              // 变量类型：.b8, .s32, .f32, etc.
    std::string space;             // 内存空间：.global, .shared, .const
    size_t alignment;              // 对齐要求
    size_t size;                   // 大小（字节）
    uint64_t address;              // 分配的内存地址
    std::vector<uint8_t> initialData;  // 初始化数据（如果有）
    
    PTXGlobalVariable() : name(""), type(""), space(""), alignment(0), size(0), address(0) {}
};

// ========================================
// 7. 符号表（全局）
// ========================================

struct PTXSymbolTable {
    // 函数符号表：函数名 → 函数定义
    std::map<std::string, PTXFunction> functions;
    
    // 全局标签：标签名 → 指令索引
    std::map<std::string, size_t> globalLabels;
    
    // 全局变量/常量：变量名 → 变量定义
    std::map<std::string, PTXGlobalVariable> variables;
    
    // 参数符号表：参数名 → 参数定义（用于快速查找）
    std::map<std::string, PTXParameter*> parameterSymbols;
    
    PTXSymbolTable() {}
    
    // 辅助方法：通过名称查找函数
    const PTXFunction* findFunction(const std::string& name) const {
        auto it = functions.find(name);
        return (it != functions.end()) ? &(it->second) : nullptr;
    }
    
    // 辅助方法：通过名称查找标签（先查局部，再查全局）
    bool findLabel(const std::string& name, const std::string& currentFunction, size_t& outIndex) const {
        // 1. 先在当前函数的局部标签中查找
        auto funcIt = functions.find(currentFunction);
        if (funcIt != functions.end()) {
            auto labelIt = funcIt->second.localLabels.find(name);
            if (labelIt != funcIt->second.localLabels.end()) {
                outIndex = labelIt->second;
                return true;
            }
        }
        
        // 2. 再在全局标签中查找
        auto globalIt = globalLabels.find(name);
        if (globalIt != globalLabels.end()) {
            outIndex = globalIt->second;
            return true;
        }
        
        return false;
    }
    
    // 辅助方法：查找参数
    const PTXParameter* findParameter(const std::string& name) const {
        auto it = parameterSymbols.find(name);
        return (it != parameterSymbols.end()) ? it->second : nullptr;
    }
};

// ========================================
// 8. 完整的 PTX 程序解析结果
// ========================================

struct PTXProgram {
    // 文件级元数据
    PTXMetadata metadata;
    
    // 所有解析的指令（线性序列）
    std::vector<DecodedInstruction> instructions;
    
    // 符号表
    PTXSymbolTable symbolTable;
    
    // 所有函数定义（包括 .entry 和 .func）
    std::vector<PTXFunction> functions;
    
    // 全局变量/常量
    std::vector<PTXGlobalVariable> globalVariables;
    
    // 入口点列表（所有 .entry 的索引）
    std::vector<size_t> entryPoints;  // 索引到 functions 数组
    
    // 错误和警告信息
    std::vector<std::string> errors;
    std::vector<std::string> warnings;
    
    PTXProgram() {}
    
    // 辅助方法：获取主入口点（第一个 .entry）
    const PTXFunction* getMainEntry() const {
        if (entryPoints.empty()) return nullptr;
        return &functions[entryPoints[0]];
    }
    
    // 辅助方法：通过名称获取入口点
    const PTXFunction* getEntryByName(const std::string& name) const {
        for (size_t idx : entryPoints) {
            if (functions[idx].name == name) {
                return &functions[idx];
            }
        }
        return nullptr;
    }
    
    // 辅助方法：打印程序结构摘要
    void printSummary() const {
        std::cout << "=== PTX Program Summary ===" << std::endl;
        std::cout << "Version: " << metadata.version << std::endl;
        std::cout << "Target: " << metadata.target << std::endl;
        std::cout << "Address Size: " << metadata.addressSize << std::endl;
        std::cout << std::endl;
        
        std::cout << "Functions: " << functions.size() << std::endl;
        for (const auto& func : functions) {
            std::cout << "  " << (func.isEntry ? "[ENTRY]" : "[FUNC ]") 
                     << " " << func.name 
                     << " (" << func.parameters.size() << " params, "
                     << (func.endInstructionIndex - func.startInstructionIndex + 1) << " instructions)"
                     << std::endl;
        }
        std::cout << std::endl;
        
        std::cout << "Total Instructions: " << instructions.size() << std::endl;
        std::cout << "Global Variables: " << globalVariables.size() << std::endl;
        std::cout << "Entry Points: " << entryPoints.size() << std::endl;
        
        if (!errors.empty()) {
            std::cout << std::endl;
            std::cout << "Errors: " << errors.size() << std::endl;
            for (const auto& err : errors) {
                std::cout << "  ERROR: " << err << std::endl;
            }
        }
        
        if (!warnings.empty()) {
            std::cout << std::endl;
            std::cout << "Warnings: " << warnings.size() << std::endl;
            for (const auto& warn : warnings) {
                std::cout << "  WARNING: " << warn << std::endl;
            }
        }
    }
};

// ========================================
// 9. 解析器接口（更新后的）
// ========================================

/*
使用示例：

PTXParser parser;
parser.parseFile("multi_function_example.ptx");

const PTXProgram& program = parser.getProgram();

// 打印摘要
program.printSummary();

// 查找并执行特定的入口点
const PTXFunction* kernel = program.getEntryByName("kernel_process_array");
if (kernel) {
    std::cout << "Found kernel: " << kernel->name << std::endl;
    std::cout << "Parameters: " << kernel->parameters.size() << std::endl;
    for (const auto& param : kernel->parameters) {
        std::cout << "  " << param.type << " " << param.name << std::endl;
    }
    
    // 执行从 kernel->startInstructionIndex 到 kernel->endInstructionIndex
}

// 获取指令序列
const auto& instructions = program.instructions;

// 解析符号引用
size_t labelAddr;
if (program.symbolTable.findLabel("loop_start", "kernel_process_array", labelAddr)) {
    std::cout << "Label 'loop_start' at instruction " << labelAddr << std::endl;
}
*/

#endif // PTX_PARSER_STRUCTURES_HPP
