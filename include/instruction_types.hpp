#ifndef INSTRUCTION_TYPES_HPP
#define INSTRUCTION_TYPES_HPP

#include <cstdint>
#include <string>
#include <vector>

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
    // Arithmetic and logic operations
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
    MOV,
    CMOV,
    
    // Parameter operations
    LD_PARAM,  // Load from parameter memory
    ST_PARAM,  // Store to parameter memory
    
    // Special operations
    NOP,
    BARRIER,
    
    // Maximum instruction type value
    MAX_INSTRUCTION_TYPE
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