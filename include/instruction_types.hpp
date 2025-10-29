#ifndef INSTRUCTION_TYPES_HPP
#define INSTRUCTION_TYPES_HPP

#include <cstdint>
#include <string>
#include <vector>
#include "memory/memory.hpp"

// Define operand types
enum class OperandType {
    REGISTER,      // Register operand
    IMMEDIATE,     // Immediate value
    MEMORY,        // Memory address
    PREDICATE,     // Predicate operand
    UNKNOWN        // Unknown operand type
};

// Instruction types
enum class InstructionTypes {
    // Arithmetic and logic operations - INTEGER
    ADD,
    SUB,
    MUL,
    DIV,
    REM, // Remainder/modulo
    AND,
    OR,
    XOR,
    NOT,
    SHL,
    SHR,
    NEG,
    ABS,
    
    // 🔧 Arithmetic operations - FLOAT
    ADD_F32, ADD_F64,
    SUB_F32, SUB_F64,
    MUL_F32, MUL_F64,
    DIV_F32, DIV_F64,
    NEG_F32, NEG_F64,
    ABS_F32, ABS_F64,
    FMA_F32, FMA_F64,     // Fused multiply-add
    SQRT_F32, SQRT_F64,   // Square root
    RSQRT_F32, RSQRT_F64, // Reciprocal square root
    MIN_F32, MIN_F64,
    MAX_F32, MAX_F64,
    
    // 🔧 Comparison and selection
    SETP,   // Set predicate based on comparison
    SELP,   // Select based on predicate
    SET,    // Set register based on comparison
    
    // 🔧 Type conversion
    CVT,    // Convert between types
    
    // Control flow instructions
    BRA,    // Branch
    JUMP,   // Jump
    CALL,   // Function call
    RET,    // Return
    
    // Synchronization instructions
    SYNC,   // Synchronization
    MEMBAR, // Memory barrier
    
    // Memory operations
    LD,
    ST,
    LD_GLOBAL,
    LD_SHARED,
    LD_LOCAL,
    ST_GLOBAL,
    ST_SHARED,
    ST_LOCAL,
    MOV,
    CMOV,
    
    // Parameter operations
    LD_PARAM,  // Load from parameter memory
    ST_PARAM,  // Store to parameter memory
    
    // 🔧 Atomic operations
    ATOM_ADD,    // Atomic add
    ATOM_SUB,    // Atomic subtract
    ATOM_EXCH,   // Atomic exchange
    ATOM_CAS,    // Atomic compare-and-swap
    ATOM_MIN,    // Atomic minimum
    ATOM_MAX,    // Atomic maximum
    ATOM_INC,    // Atomic increment
    ATOM_DEC,    // Atomic decrement
    ATOM_AND,    // Atomic AND
    ATOM_OR,     // Atomic OR
    ATOM_XOR,    // Atomic XOR
    
    // Special operations
    NOP,
    BARRIER,
    
    // Maximum instruction type value
    MAX_INSTRUCTION_TYPE
};

// 🔧 比较操作符
enum class CompareOp {
    EQ,  // Equal
    NE,  // Not equal
    LT,  // Less than
    LE,  // Less than or equal
    GT,  // Greater than
    GE,  // Greater than or equal
    LO,  // Lower (unsigned)
    LS,  // Lower or same (unsigned)
    HI,  // Higher (unsigned)
    HS   // Higher or same (unsigned)
};

// 🔧 数据类型
enum class DataType {
    S8, S16, S32, S64,   // Signed integers
    U8, U16, U32, U64,   // Unsigned integers
    F16, F32, F64,       // Floating point
    B8, B16, B32, B64    // Bit-type (untyped)
};

// Synchronization type
enum class SyncType {
    SYNC_UNDEFINED,  // Undefined or unsupported
    SYNC_WARP,       // Warp-level synchronization
    SYNC_CTA,        // CTA-level synchronization
    SYNC_GRID,       // Grid-level synchronization
    SYNC_MEMBAR      // Memory barrier
};

// Define operand structure
struct Operand {
    OperandType type;              // Type of operand
    union {
        uint32_t registerIndex;    // For REGISTER type
        int64_t immediateValue;    // For IMMEDIATE type
        uint64_t address;          // For MEMORY type
        uint32_t predicateIndex;   // For PREDICATE type
    };
    
    // Additional flags
    bool isAddress;                // Is this an address operand?
    bool isIndirect;               // Is this an indirect access?
};

// Define decoded instruction structure
struct DecodedInstruction {
    InstructionTypes type;           // Instruction type (from InstructionTypes)
    Operand dest;                   // Destination operand
    std::vector<Operand> sources;   // Source operands
    uint32_t modifiers;             // Instruction modifiers
    bool hasPredicate;              // Does this instruction have a predicate?
    uint32_t predicateIndex;        // Index of the predicate register
    bool predicateValue;            // Value of the predicate (true/false)
    
    // 🔧 Additional fields for new instructions
    CompareOp compareOp = CompareOp::EQ;  // For SETP, SET instructions
    DataType dataType = DataType::S32;    // Data type for operation
    DataType srcType = DataType::S32;     // Source type for CVT
    DataType dstType = DataType::S32;     // Destination type for CVT
    MemorySpace memorySpace = MemorySpace::GLOBAL; // For ATOM instructions
};

// Define PTX instruction structure
struct PTXInstruction {
    std::string opcode;             // Instruction opcode (e.g., "add", "mov", "ld")
    std::string predicate;          // Predicate register (e.g., "p0", empty if none)
    std::string dest;               // Destination operand
    std::vector<std::string> sources; // Source operands
    std::vector<std::string> modifiers; // Instruction modifiers
};

#endif // INSTRUCTION_TYPES_HPP