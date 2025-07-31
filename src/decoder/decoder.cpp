#include "decoder.hpp"
#include <unordered_map>
#include <cctype>

// Private implementation class to hide decoding details
class PTXDecoder::Impl {
public:
    Impl(VMCore* vmCore) : m_vmCore(vmCore) {}
    
    // Decode a collection of PTX instructions
    bool decodeInstructions(const std::vector<PTXInstruction>& ptInstructions);
    
    // Get decoded instruction count
    size_t getDecodedInstructionCount() const {
        return m_decodedInstructions.size();
    }
    
private:
    VMCore* m_vmCore;
    std::vector<DecodedInstruction> m_decodedInstructions;
    
    // Map of PTX opcodes to internal instruction types
    std::unordered_map<std::string, InstructionType> m_opcodeMap;
    
    // Initialize the opcode mapping
    void initializeOpcodeMap();
    
    // Decode a single PTX instruction
    bool decodeSingleInstruction(const PTXInstruction& ptInstr, DecodedInstruction& outDecodedInstr);
    
    // Helper functions for decoding
    bool isRegister(const std::string& operand);
    RegisterIndex parseRegister(const std::string& regStr);
    bool isImmediate(const std::string& operand);
    ImmediateValue parseImmediate(const std::string& immStr);
    bool isMemoryAccess(const std::string& operand);
    uint64_t parseMemoryAddress(const std::string& addrStr);
};

PTXDecoder::PTXDecoder(VMCore* vmCore) : pImpl(std::make_unique<Impl>(vmCore)) {}

PTXDecoder::~PTXDecoder() = default;

bool PTXDecoder::decodeInstructions(const std::vector<PTXInstruction>& ptInstructions) {
    return pImpl->decodeInstructions(ptInstructions);
}

size_t PTXDecoder::getDecodedInstructionCount() const {
    return pImpl->getDecodedInstructionCount();
}

bool PTXDecoder::Impl::decodeInstructions(const std::vector<PTXInstruction>& ptInstructions) {
    m_decodedInstructions.clear();
    m_decodedInstructions.reserve(ptInstructions.size());
    
    for (const auto& ptInstr : ptInstructions) {
        DecodedInstruction decodedInstr;
        if (decodeSingleInstruction(ptInstr, decodedInstr)) {
            m_decodedInstructions.push_back(decodedInstr);
        }
    }
    
    return true;
}

// No need to modify the DecodedInstruction struct as it's already defined appropriately

void PTXDecoder::Impl::initializeOpcodeMap() {
    // Initialize mapping from PTX opcodes to internal instruction types
    m_opcodeMap["add"] = InstructionTypes::ADD;
    m_opcodeMap["sub"] = InstructionTypes::SUB;
    m_opcodeMap["mul"] = InstructionTypes::MUL;
    m_opcodeMap["mad"] = InstructionTypes::MUL;  // mad uses same type as mul
    m_opcodeMap["div"] = InstructionTypes::DIV;
    m_opcodeMap["and"] = InstructionTypes::AND;
    m_opcodeMap["or"] = InstructionTypes::OR;
    m_opcodeMap["xor"] = InstructionTypes::XOR;
    m_opcodeMap["not"] = InstructionTypes::NOT;
    m_opcodeMap["shl"] = InstructionTypes::SHL;
    m_opcodeMap["shr"] = InstructionTypes::SHR;
    m_opcodeMap["neg"] = InstructionTypes::NEG;
    m_opcodeMap["abs"] = InstructionTypes::ABS;
    m_opcodeMap["bra"] = InstructionTypes::BRA;
    m_opcodeMap["exit"] = InstructionTypes::EXIT;
    m_opcodeMap["ret"] = InstructionTypes::RETURN;
    m_opcodeMap["call"] = InstructionTypes::CALL;
    m_opcodeMap["ld"] = InstructionTypes::LD;
    m_opcodeMap["st"] = InstructionTypes::ST;
    m_opcodeMap["mov"] = InstructionTypes::MOV;
    m_opcodeMap["cmov"] = InstructionTypes::CMOV;
    m_opcodeMap["nop"] = InstructionTypes::NOP;
    m_opcodeMap["barrier"] = InstructionTypes::BARRIER;
    m_opcodeMap["sync"] = InstructionTypes::SYNC;
    // ... add more opcode mappings as needed ...
}

bool PTXDecoder::Impl::decodeSingleInstruction(const PTXInstruction& ptInstr, DecodedInstruction& outDecodedInstr) {
    // First ensure our opcode map is initialized
    if (m_opcodeMap.empty()) {
        initializeOpcodeMap();
    }
    
    // Look up the instruction type
    auto it = m_opcodeMap.find(ptInstr.opcode);
    if (it == m_opcodeMap.end()) {
        // Unknown opcode
        outDecodedInstr.type = InstructionTypes::INVALID;
        return false;
    }
    
    outDecodedInstr.type = it->second;
    
    // Handle predicate
    if (!ptInstr.predicate.empty()) {
        // Parse predicate information
        // This is a simplified example - actual implementation would be more complex
        outDecodedInstr.hasPredicate = true;
        outDecodedInstr.predicateIndex = parseRegister(ptInstr.predicate);
        // In real implementation, we would read the predicate value from register
        outDecodedInstr.predicateValue = true;  // Simplified for now
    } else {
        outDecodedInstr.hasPredicate = false;
    }
    
    // Handle destination operand
    if (!ptInstr.dest.empty()) {
        // Parse destination register
        // In a real implementation, this would involve register allocation
        outDecodedInstr.dest.type = OperandType::REGISTER;
        outDecodedInstr.dest.registerIndex = parseRegister(ptInstr.dest);
        outDecodedInstr.dest.isAddress = false;
        outDecodedInstr.dest.isIndirect = false;
    }
    
    // Handle source operands
    for (const auto& source : ptInstr.sources) {
        Operand decodedOperand;
        
        if (isRegister(source)) {
            decodedOperand.type = OperandType::REGISTER;
            decodedOperand.registerIndex = parseRegister(source);
            decodedOperand.isAddress = false;
            decodedOperand.isIndirect = false;
        } else if (isImmediate(source)) {
            decodedOperand.type = OperandType::IMMEDIATE;
            decodedOperand.immediateValue = parseImmediate(source);
            decodedOperand.isAddress = false;
            decodedOperand.isIndirect = false;
        } else if (isMemoryAccess(source)) {
            decodedOperand.type = OperandType::MEMORY;
            decodedOperand.address = parseMemoryAddress(source);
            decodedOperand.isAddress = true;
            decodedOperand.isIndirect = (source.find('*') != std::string::npos);
        } else {
            // Handle other operand types
            decodedOperand.type = OperandType::UNKNOWN;
        }
        
        outDecodedInstr.sources.push_back(decodedOperand);
    }
    
    // Handle instruction modifiers
    for (const auto& modifier : ptInstr.modifiers) {
        // Parse and store modifiers
        // For now, just ignore them
        (void)modifier;
    }
    
    return true;
}

// Helper functions (simplified)
bool PTXDecoder::Impl::isRegister(const std::string& operand) {
    return !operand.empty() && operand[0] == '%';
}

RegisterIndex PTXDecoder::Impl::parseRegister(const std::string& regStr) {
    // Simplified register parsing
    // In a real implementation, this would handle different register types,
    // allocate virtual registers, etc.
    static std::unordered_map<std::string, RegisterIndex> registerMap;
    
    // Lazy initialization of register map
    if (registerMap.empty()) {
        // Initialize with some standard registers
        // ...
    }
    
    // Create/register the register if not found
    auto it = registerMap.find(regStr);
    if (it == registerMap.end()) {
        static RegisterIndex nextIndex = 0;
        it = registerMap.emplace(regStr, nextIndex++).first;
    }
    
    return it->second;
}

bool PTXDecoder::Impl::isImmediate(const std::string& operand) {
    // Check if operand is an immediate value
    return !operand.empty() && (std::isdigit(operand[0]) || operand[0] == '#' || operand[0] == '-' || operand[0] == '0');
}

ImmediateValue PTXDecoder::Impl::parseImmediate(const std::string& immStr) {
    // Parse immediate value (simplified)
    ImmediateValue result = 0;
    if (immStr[0] == '#') {
        // Hex value
        sscanf_s(immStr.c_str(), "#%lx", &result);
    } else if (immStr.substr(0, 2) == "0x") {
        // Hex value with 0x prefix
        sscanf_s(immStr.c_str(), "%lx", &result);
    } else {
        // Decimal value
        result = static_cast<ImmediateValue>(std::atol(immStr.c_str()));
    }
    return result;
}

bool PTXDecoder::Impl::isMemoryAccess(const std::string& operand) {
    // Simple check for memory access syntax (e.g., [address], [*address])
    return operand.find('[') != std::string::npos || operand.find('*') != std::string::npos;
}

uint64_t PTXDecoder::Impl::parseMemoryAddress(const std::string& addrStr) {
    // Simplified memory address parsing
    // In a real implementation, this would handle complex addressing modes
    uint64_t address = 0;
    
    // Extract address expression between brackets if present
    size_t start = addrStr.find('[');
    if (start != std::string::npos) {
        size_t end = addrStr.find(']', start + 1);
        if (end != std::string::npos) {
            std::string expr = addrStr.substr(start + 1, end - start - 1);
            // For simplicity, assume direct register addressing
            if (expr[0] == '%') {
                // Address is in a register
                RegisterIndex regIndex = parseRegister(expr);
                // In real implementation, we would read the register value here
                (void)regIndex; // Unused in this simplified version
            } else {
                // Constant address
                address = static_cast<uint64_t>(std::atoll(expr.c_str()));
            }
        }
    } else {
        // Simple register indirect addressing
        if (addrStr[0] == '*') {
            // Address is in a register that follows
            if (addrStr.size() > 1 && addrStr[1] == '%') {
                std::string regStr = addrStr.substr(1);
                RegisterIndex regIndex = parseRegister(regStr);
                // In real implementation, we would read the register value here
                (void)regIndex; // Unused in this simplified version
            }
        } else if (addrStr[0] == '%') {
            // Address is in a register
            RegisterIndex regIndex = parseRegister(addrStr);
            // In real implementation, we would read the register value here
            (void)regIndex; // Unused in this simplified version
        } else {
            // Constant address
            address = static_cast<uint64_t>(std::atoll(addrStr.c_str()));
        }
    }
    
    return address;
}

// Decode an instruction from PTX code
bool Decoder::decodeInstruction(const std::string& instruction, DecodedInstruction& decoded) {
    // Skip empty lines
    if (instruction.empty()) {
        return false;
    }
    
    // Skip comments
    if (instruction.find("//") == 0 || instruction.find("/*") == 0) {
        return false;
    }
    
    // Reset decoded instruction
    decoded = DecodedInstruction();
    
    // Parse instruction
    std::string trimmed = trim(instruction);
    
    // Check instruction type
    if (trimmed.find("add") == 0) {
        return decodeArithmeticInstruction(trimmed, decoded);
    } else if (trimmed.find("mul") == 0 || trimmed.find("mad") == 0) {
        return decodeArithmeticInstruction(trimmed, decoded);
    } else if (trimmed.find("bra") == 0) {
        return decodeBranchInstruction(trimmed, decoded);
    } else if (trimmed.find("ld") == 0 || trimmed.find("st") == 0) {
        return decodeLoadStoreInstruction(trimmed, decoded);
    } else if (trimmed.find("bar.sync") == 0 || trimmed.find("bar.arrive") == 0) {
        return decodeSynchronizationInstruction(trimmed, decoded);
    } else if (trimmed.find("membar") == 0) {
        return decodeMemoryBarrierInstruction(trimmed, decoded);
    } else if (trimmed.find("@") == 1 && (trimmed.find("bra") == 3 || trimmed.find("bra.uni") == 3)) {
        // Predicate branch instruction
        return decodeBranchInstruction(trimmed, decoded);
    }
    
    // Default to unknown instruction
    m_logger.log(LogLevel::WARNING, "Unknown instruction: " + instruction);
    return false;
}

// Decode synchronization instruction
bool Decoder::decodeSynchronizationInstruction(const std::string& instruction, DecodedInstruction& decoded) {
    // Initialize decoded instruction
    decoded.type = InstructionType::SYNC;
    
    // Parse instruction modifiers
    std::istringstream iss(instruction);
    std::string token;
    
    while (iss >> token) {
        // Skip the instruction mnemonic
        if (token == "bar.sync" || token == "bar.arrive" || token == "membar") {
            continue;
        }
        
        // Parse synchronization type
        if (token == "cta") {
            decoded.syncType = SyncType::SYNC_CTA;
        } else if (token == "warp") {
            decoded.syncType = SyncType::SYNC_WARP;
        } else if (token == "grid") {
            decoded.syncType = SyncType::SYNC_GRID;
        } else if (token == "global" || token == "sys") {
            decoded.syncType = SyncType::SYNC_MEMBAR;
        } else if (token == "uni") {
            // Uniform execution mode
            decoded.flags |= DECODE_FLAG_UNIFORM;
        }
        
        // Parse predicate
        if (token.size() > 0 && token[0] == '@') {
            // Predicate found
            decoded.hasPredicate = true;
            
            // Extract predicate ID
            std::string predStr = token.substr(1);  // Remove '@' prefix
            if (predStr.size() > 0 && predStr[0] == '!') {
                predStr = predStr.substr(1);  // Remove '!' if present
                decoded.flags |= DECODE_FLAG_NEGATE_PREDICATE;
            }
            
            // Convert to numeric ID
            try {
                decoded.predicateId = static_cast<uint32_t>(std::stoul(predStr));
            } catch (...) {
                m_logger.log(LogLevel::WARNING, "Invalid predicate ID: " + predStr);
                return false;
            }
        }
    }
    
    return true;
}

// Decode memory barrier instruction
bool Decoder::decodeMemoryBarrierInstruction(const std::string& instruction, DecodedInstruction& decoded) {
    // Initialize decoded instruction
    decoded.type = InstructionType::MEMBAR;
    
    // Set default synchronization type
    decoded.syncType = SyncType::SYNC_MEMBAR;
    
    // Parse instruction modifiers
    std::istringstream iss(instruction);
    std::string token;
    
    while (iss >> token) {
        // Skip the instruction mnemonic
        if (token == "membar") {
            continue;
        }
        
        // Parse memory scope
        if (token == "global" || token == "sys") {
            decoded.syncType = SyncType::SYNC_MEMBAR;
        } else if (token == "cta") {
            decoded.syncType = SyncType::SYNC_CTA;
        } else if (token == "warp") {
            decoded.syncType = SyncType::SYNC_WARP;
        } else if (token == "grid") {
            decoded.syncType = SyncType::SYNC_GRID;
        } else if (token == "uni") {
            // Uniform execution mode
            decoded.flags |= DECODE_FLAG_UNIFORM;
        }
        
        // Parse predicate
        if (token.size() > 0 && token[0] == '@') {
            // Predicate found
            decoded.hasPredicate = true;
            
            // Extract predicate ID
            std::string predStr = token.substr(1);  // Remove '@' prefix
            if (predStr.size() > 0 && predStr[0] == '!') {
                predStr = predStr.substr(1);  // Remove '!' if present
                decoded.flags |= DECODE_FLAG_NEGATE_PREDICATE;
            }
            
            // Convert to numeric ID
            try {
                decoded.predicateId = static_cast<uint32_t>(std::stoul(predStr));
            } catch (...) {
                m_logger.log(LogLevel::WARNING, "Invalid predicate ID: " + predStr);
                return false;
            }
        }
    }
    
    return true;
}
