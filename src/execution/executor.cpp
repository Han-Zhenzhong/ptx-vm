#include "executor.hpp"
#include <stdexcept>
#include <iostream>
#include <sstream>
#include <cmath>
#include <algorithm>
#include "performance_counters.hpp"
#include "warp_scheduler.hpp"  // Warp scheduler header
#include "predicate_handler.hpp"  // Predicate handler header
#include "reconvergence_mechanism.hpp"  // Reconvergence mechanism header
#include "memory/memory.hpp"  // Memory subsystem and MemorySpace

// Private implementation class
class PTXExecutor::Impl {
public:
    Impl() : m_registerBank(nullptr), m_memorySubsystem(nullptr), m_performanceCounters(nullptr) {
        // Note: RegisterBank and MemorySubsystem will be set via setComponents()
        // They are owned externally (by PTXVM)
        
        // Initialize warp scheduler with single warp for now
        // TODO: Support multiple warps with per-thread register files
        m_warpScheduler = std::make_unique<WarpScheduler>(1, 32);
        if (!m_warpScheduler->initialize()) {
            throw std::runtime_error("Failed to initialize warp scheduler");
        }

        // Initialize predicate handler
        m_predicateHandler = std::make_unique<PredicateHandler>();
        if (!m_predicateHandler->initialize()) {
            throw std::runtime_error("Failed to initialize predicate handler");
        }

        // Set execution mode to SIMT
        m_predicateHandler->setExecutionMode(EXECUTION_MODE_SIMT);

        // Initialize reconvergence mechanism with CFG-based algorithm
        m_reconvergence = std::make_unique<ReconvergenceMechanism>();
        if (!m_reconvergence->initialize(RECONVERGENCE_ALGORITHM_CFG_BASED)) {
            throw std::runtime_error("Failed to initialize reconvergence mechanism");
        }
    }
    
    ~Impl() = default;

    // Initialize decoded instructions
    bool initialize(const std::vector<PTXInstruction>& ptInstructions) {
        m_ptInstructions = ptInstructions;
        
        // Initialize decoder
        m_decoder = std::make_unique<Decoder>(nullptr);
        if (!m_decoder->decodeInstructions(m_ptInstructions)) {
            return false;
        }
        
        m_decodedInstructions = m_decoder->getDecodedInstructions();
        m_currentInstructionIndex = 0;
        m_executionComplete = false;

        // Build control flow graph from decoded instructions
        std::vector<std::vector<size_t>> cfg;
        buildCFG(m_decodedInstructions, cfg);
        
        // Set the control flow graph in the reconvergence mechanism
        m_reconvergence->setControlFlowGraph(cfg);
        
        return true;
    }

    // Set decoded instructions directly
    void setDecodedInstructions(const std::vector<DecodedInstruction>& decodedInstructions) {
        m_decodedInstructions = decodedInstructions;
    }
    
    // Set current instruction index
    void setCurrentInstructionIndex(size_t index) {
        m_currentInstructionIndex = index;
    }
    
    // Set execution complete flag
    void setExecutionComplete(bool complete) {
        m_executionComplete = complete;
    }
    
    // Execute all instructions
    bool execute() {
        if (m_decodedInstructions.empty() || m_executionComplete) {
            std::cout << "No instructions to execute" << std::endl;
            return false;
        }
        m_performanceCounters->increment(PerformanceCounterIDs::INSTRUCTIONS_EXECUTED, m_decodedInstructions.size());
        size_t numWarps = m_warpScheduler->getNumWarps();
        std::vector<bool> warpDone(numWarps, false);
        size_t doneWarps = 0;
        while (doneWarps < numWarps) {
            m_performanceCounters->increment(PerformanceCounterIDs::CYCLES);
            for (uint32_t warpId = 0; warpId < numWarps; ++warpId) {
                if (warpDone[warpId]) continue;
                uint64_t activeMask = m_warpScheduler->getActiveThreads(warpId);
                if (activeMask == 0) {
                    warpDone[warpId] = true;
                    ++doneWarps;
                    continue;
                }
                InstructionIssueInfo issueInfo;
                if (!m_warpScheduler->issueInstruction(issueInfo)) {
                    // No instruction to issue, mark as done
                    warpDone[warpId] = true;
                    ++doneWarps;
                    continue;
                }
                if (issueInfo.instructionIndex >= m_decodedInstructions.size()) {
                    warpDone[warpId] = true;
                    ++doneWarps;
                    continue;
                }
                const DecodedInstruction& instr = m_decodedInstructions[issueInfo.instructionIndex];
                bool shouldExecute = m_predicateHandler->shouldExecute(instr);
                if (shouldExecute) {
                    // ✅ Set m_currentInstructionIndex to current PC before execution
                    // This is needed for instructions that modify it (branch, call, ret)
                    m_currentInstructionIndex = issueInfo.instructionIndex;
                    size_t pcBefore = m_currentInstructionIndex;
                    
                    bool result = executeDecodedInstruction(instr);
                    if (!result) {
                        std::cout << "Error executing instruction" << std::endl;
                        return false;
                    }
                    
                    // ✅ Check if instruction modified PC (branch, call, ret)
                    if (m_currentInstructionIndex != pcBefore) {
                        // Instruction changed PC - sync to warp (don't auto-increment)
                        m_warpScheduler->setCurrentPC(warpId, m_currentInstructionIndex);
                    } else {
                        // Normal instruction - let warp auto-increment PC
                        m_warpScheduler->completeInstruction(issueInfo);
                    }
                    
                    // ✅ Check if execution is complete (from ret instruction)
                    if (m_executionComplete) {
                        warpDone[warpId] = true;
                        ++doneWarps;
                    }
                } else {
                    m_performanceCounters->increment(PerformanceCounterIDs::PREDICATE_SKIPPED);
                    // 跳过时也推进PC
                    m_warpScheduler->completeInstruction(issueInfo);
                }
            }
        }
        m_executionComplete = true;
        return true;
    }

    // Get current instruction index
    size_t getCurrentInstructionIndex() const {
        return m_currentInstructionIndex;
    }

    // Check if execution is complete
    bool isExecutionComplete() const {
        return m_executionComplete;
    }

    // Get decoded instructions for debugging
    const std::vector<DecodedInstruction>& getDecodedInstructions() const {
        return m_decodedInstructions;
    }

    // Get references to core components
    RegisterBank& getRegisterBank() {
        return *m_registerBank;
    }
    
    MemorySubsystem& getMemorySubsystem() {
        return *m_memorySubsystem;
    }

    WarpScheduler& getWarpScheduler() {
        return *m_warpScheduler;
    }

    // Get reference to performance counters
    PerformanceCounters& getPerformanceCounters() {
        return *m_performanceCounters;
    }

    // Build control flow graph from decoded instructions
    void buildCFGFromDecodedInstructions(const std::vector<DecodedInstruction>& decodedInstructions) {
        std::vector<std::vector<size_t>> cfg;
        buildCFG(decodedInstructions, cfg);
        
        // Set the control flow graph in the reconvergence mechanism
        m_reconvergence->setControlFlowGraph(cfg);
    }

    // Set register bank and memory subsystem (from external sources)
    void setComponents(RegisterBank& registerBank, MemorySubsystem& memorySubsystem) {
        // Use the externally provided components (owned by PTXVM)
        m_registerBank = &registerBank;
        m_memorySubsystem = &memorySubsystem;
    }

    void setPerformanceCounters(PerformanceCounters& performanceCounters)
    {
        m_performanceCounters = &performanceCounters;
    }

    // Execute a single instruction
    bool executeSingleInstruction() {
        if (m_currentInstructionIndex >= m_decodedInstructions.size()) {
            return false;
        }
        
        // Get the current instruction
        const auto& instr = m_decodedInstructions[m_currentInstructionIndex];
        
        // Increment cycle counter
        m_performanceCounters->increment(PerformanceCounterIDs::CYCLES);
        
        // Execute the instruction
        bool result = executeDecodedInstruction(instr);
        
        // Increment instruction counter
        m_performanceCounters->increment(PerformanceCounterIDs::INSTRUCTIONS_EXECUTED);
        
        return result;
    }
    
    // Execute LD for a specific memory space
    bool executeLDMemorySpace(const DecodedInstruction& instr, MemorySpace space) {
        if (instr.dest.type != OperandType::REGISTER || instr.sources.size() != 1 ||
            instr.sources[0].type != OperandType::MEMORY) {
            std::cerr << "Invalid LD instruction format" << std::endl;
            m_currentInstructionIndex++;
            return true;
        }
        
        // Calculate memory address
        uint64_t address = instr.sources[0].address;
        
        // Handle register-indirect addressing [%rX] or [%rX+offset]
        if (instr.sources[0].isIndirect) {
            // Get base address from register (use baseRegisterIndex for memory operands)
            uint64_t baseAddr = m_registerBank->readRegister(instr.sources[0].baseRegisterIndex);
            address = baseAddr + instr.sources[0].address;  // Add offset if any
        }
        
        // Increment appropriate memory read counter
        switch (space) {
            case MemorySpace::GLOBAL:
                m_performanceCounters->increment(PerformanceCounterIDs::GLOBAL_MEMORY_READS);
                break;
            case MemorySpace::SHARED:
                m_performanceCounters->increment(PerformanceCounterIDs::SHARED_MEMORY_READS);
                break;
            case MemorySpace::LOCAL:
                m_performanceCounters->increment(PerformanceCounterIDs::LOCAL_MEMORY_READS);
                break;
            case MemorySpace::PARAMETER:
                m_performanceCounters->increment(PerformanceCounterIDs::PARAMETER_MEMORY_READS);
                break;
            default:
                break;
        }
        
        // Read from memory based on data type
        uint64_t value = 0;
        switch (instr.dataType) {
            case DataType::S8:
                // Read as unsigned, then sign-extend
                value = static_cast<uint64_t>(static_cast<int64_t>(static_cast<int8_t>(m_memorySubsystem->read<uint8_t>(space, address))));
                break;
            case DataType::U8:
                value = static_cast<uint64_t>(m_memorySubsystem->read<uint8_t>(space, address));
                break;
            case DataType::S16:
                // Read as unsigned, then sign-extend
                value = static_cast<uint64_t>(static_cast<int64_t>(static_cast<int16_t>(m_memorySubsystem->read<uint16_t>(space, address))));
                break;
            case DataType::U16:
                value = static_cast<uint64_t>(m_memorySubsystem->read<uint16_t>(space, address));
                break;
            case DataType::S32:
                // Read as unsigned, then sign-extend
                value = static_cast<uint64_t>(static_cast<int64_t>(static_cast<int32_t>(m_memorySubsystem->read<uint32_t>(space, address))));
                break;
            case DataType::U32:
            case DataType::F32:
                value = static_cast<uint64_t>(m_memorySubsystem->read<uint32_t>(space, address));
                break;
            case DataType::S64:
            case DataType::U64:
            case DataType::F64:
            default:
                value = m_memorySubsystem->read<uint64_t>(space, address);
                break;
        }
        
        storeRegisterValue(instr.dest.registerIndex, value);
        m_currentInstructionIndex++;
        return true;
    }

    // Execute ST for a specific memory space
    bool executeSTMemorySpace(const DecodedInstruction& instr, MemorySpace space) {
        // New convention: dest is memory address, sources[0] is data
        if (instr.dest.type != OperandType::MEMORY || instr.sources.size() != 1) {
            std::cerr << "Invalid ST instruction format" << std::endl;
            m_currentInstructionIndex++;
            return true;
        }
        
        int64_t src = getSourceValue(instr.sources[0]);
        
        // Calculate memory address
        uint64_t address = instr.dest.address;
        
        // Handle register-indirect addressing [%rX] or [%rX+offset]
        if (instr.dest.isIndirect) {
            // Get base address from register (use baseRegisterIndex for memory operands)
            uint64_t baseAddr = m_registerBank->readRegister(instr.dest.baseRegisterIndex);
            std::cout << "ST: Base address from %r" << instr.dest.baseRegisterIndex 
                      << " = 0x" << std::hex << baseAddr 
                      << ", offset = 0x" << instr.dest.address << std::dec << std::endl;
            address = baseAddr + instr.dest.address;  // Add offset if any
            std::cout << "ST: Final address = 0x" << std::hex << address << std::dec << std::endl;
        }
        
        // Increment appropriate memory write counter
        switch (space) {
            case MemorySpace::GLOBAL:
                m_performanceCounters->increment(PerformanceCounterIDs::GLOBAL_MEMORY_WRITES);
                break;
            case MemorySpace::SHARED:
                m_performanceCounters->increment(PerformanceCounterIDs::SHARED_MEMORY_WRITES);
                break;
            case MemorySpace::LOCAL:
                m_performanceCounters->increment(PerformanceCounterIDs::LOCAL_MEMORY_WRITES);
                break;
            case MemorySpace::PARAMETER:
                m_performanceCounters->increment(PerformanceCounterIDs::PARAMETER_MEMORY_WRITES);
                break;
            default:
                break;
        }
        
        // Write to memory based on data type
        switch (instr.dataType) {
            case DataType::S8:
            case DataType::U8:
                m_memorySubsystem->write<uint8_t>(space, address, static_cast<uint8_t>(src));
                break;
            case DataType::S16:
            case DataType::U16:
                m_memorySubsystem->write<uint16_t>(space, address, static_cast<uint16_t>(src));
                break;
            case DataType::S32:
            case DataType::U32:
            case DataType::F32:
                m_memorySubsystem->write<uint32_t>(space, address, static_cast<uint32_t>(src));
                break;
            case DataType::S64:
            case DataType::U64:
            case DataType::F64:
            default:
                m_memorySubsystem->write<uint64_t>(space, address, static_cast<uint64_t>(src));
                break;
        }
        
        m_currentInstructionIndex++;
        return true;
    }
    // Execute a decoded instruction
    bool executeDecodedInstruction(const DecodedInstruction& instr) {
        // Check if instruction has predicate and should be skipped
        if (instr.hasPredicate) {
            // Read predicate value - can be from predicate register or integer register
            // PTX allows using integer registers as predicates (0 = false, non-zero = true)
            bool predicateRegValue;
            
            // Check if this is within predicate register range (typically 0-7 or 0-15)
            // If beyond that, it's likely stored in a regular register
            if (instr.predicateIndex < 16) {
                // Try to read as predicate register
                try {
                    predicateRegValue = m_registerBank->readPredicate(instr.predicateIndex);
                } catch (const std::out_of_range&) {
                    // If out of range, read from regular register instead
                    uint64_t regValue = m_registerBank->readRegister(instr.predicateIndex);
                    predicateRegValue = (regValue != 0);
                }
            } else {
                // Definitely a regular register
                uint64_t regValue = m_registerBank->readRegister(instr.predicateIndex);
                predicateRegValue = (regValue != 0);
            }
            
            // instr.predicateValue indicates if predicate is negated:
            // - true (not negated): execute if predicate register is true
            // - false (negated): execute if predicate register is false
            bool shouldExecute = (instr.predicateValue == predicateRegValue);
            
            if (!shouldExecute) {
                // Skip this instruction
                m_currentInstructionIndex++;
                return true;
            }
        }
        
        // Dispatch based on instruction type
        switch (instr.type) {
            case InstructionTypes::ADD:
                return executeADD(instr);
            case InstructionTypes::SUB:
                return executeSUB(instr);
            case InstructionTypes::MUL:
                return executeMUL(instr);
            case InstructionTypes::DIV:
                return executeDIV(instr);
            case InstructionTypes::REM:
                return executeREM(instr);
            case InstructionTypes::AND:
                return executeAND(instr);
            case InstructionTypes::OR:
                return executeOR(instr);
            case InstructionTypes::XOR:
                return executeXOR(instr);
            case InstructionTypes::NOT:
                return executeNOT(instr);
            case InstructionTypes::SHL:
                return executeSHL(instr);
            case InstructionTypes::SHR:
                return executeSHR(instr);
            case InstructionTypes::NEG:
                return executeNEG(instr);
            case InstructionTypes::ABS:
                return executeABS(instr);
            case InstructionTypes::MOV:
                return executeMOV(instr);
            case InstructionTypes::LD:
                return executeLD(instr);
            case InstructionTypes::ST:
                return executeST(instr);
            case InstructionTypes::LD_GLOBAL:
                return executeLDMemorySpace(instr, MemorySpace::GLOBAL);
            case InstructionTypes::LD_SHARED:
                return executeLDMemorySpace(instr, MemorySpace::SHARED);
            case InstructionTypes::LD_LOCAL:
                return executeLDMemorySpace(instr, MemorySpace::LOCAL);
            case InstructionTypes::LD_PARAM:
                return executeLDParam(*this, instr);
            case InstructionTypes::ST_GLOBAL:
                return executeSTMemorySpace(instr, MemorySpace::GLOBAL);
            case InstructionTypes::ST_SHARED:
                return executeSTMemorySpace(instr, MemorySpace::SHARED);
            case InstructionTypes::ST_LOCAL:
                return executeSTMemorySpace(instr, MemorySpace::LOCAL);
            case InstructionTypes::ST_PARAM:
                return executeSTParam(*this, instr);
            case InstructionTypes::BRA:
                return executeBRA(instr);
            case InstructionTypes::JUMP:
                return executeJUMP(instr);
            case InstructionTypes::CALL:
                return executeCALL(instr);
            case InstructionTypes::RET:
                return executeEXIT(instr);
            case InstructionTypes::NOP:
                return executeNOP(instr);
            case InstructionTypes::CMOV:
                return executeCMOV(instr);
            case InstructionTypes::SYNC:
                return executeSYNC(instr);
            case InstructionTypes::MEMBAR:
                return executeMEMBAR(instr);
            case InstructionTypes::BARRIER:
                return executeBARRIER(instr);
            
            // Floating-point instructions
            case InstructionTypes::ADD_F32:
                return executeADD_F32(instr);
            case InstructionTypes::SUB_F32:
                return executeSUB_F32(instr);
            case InstructionTypes::MUL_F32:
                return executeMUL_F32(instr);
            case InstructionTypes::DIV_F32:
                return executeDIV_F32(instr);
            case InstructionTypes::FMA_F32:
                return executeFMA_F32(instr);
            case InstructionTypes::SQRT_F32:
                return executeSQRT_F32(instr);
            case InstructionTypes::NEG_F32:
                return executeNEG_F32(instr);
            case InstructionTypes::ABS_F32:
                return executeABS_F32(instr);
            
            // Comparison and selection instructions
            case InstructionTypes::SETP:
                return executeSETP(instr);
            case InstructionTypes::SELP:
                return executeSELP(instr);
            
            // Type conversion instructions
            case InstructionTypes::CVT:
                return executeCVT(instr);
            
            // Atomic operations
            case InstructionTypes::ATOM_ADD:
                return executeATOM_ADD(instr);
            case InstructionTypes::ATOM_SUB:
                return executeATOM_SUB(instr);
            case InstructionTypes::ATOM_EXCH:
                return executeATOM_EXCH(instr);
            case InstructionTypes::ATOM_CAS:
                return executeATOM_CAS(instr);
            case InstructionTypes::ATOM_MIN:
                return executeATOM_MIN(instr);
            case InstructionTypes::ATOM_MAX:
                return executeATOM_MAX(instr);
            
            default:
                std::cerr << "Unsupported instruction type: " << static_cast<int>(instr.type) << std::endl;
                m_currentInstructionIndex++;
                return true; // Continue execution
        }
    }

    // Execute REM (remainder) instruction
    bool executeREM(const DecodedInstruction& instr) {
        if (instr.dest.type != OperandType::REGISTER || instr.sources.size() != 2) {
            std::cerr << "Invalid REM instruction format" << std::endl;
            m_currentInstructionIndex++;
            return true;
        }
        int64_t src0 = getSourceValue(instr.sources[0]);
        int64_t src1 = getSourceValue(instr.sources[1]);
        if (src1 == 0) {
            std::cerr << "Division by zero in REM" << std::endl;
            m_currentInstructionIndex++;
            return true;
        }
        int64_t result = src0 % src1;
        storeRegisterValue(instr.dest.registerIndex, static_cast<uint64_t>(result));
        m_currentInstructionIndex++;
        return true;
    }

    // --- 新增指令类型的执行函数 ---
    bool executeAND(const DecodedInstruction& instr) {
        if (instr.dest.type != OperandType::REGISTER || instr.sources.size() != 2) {
            std::cerr << "Invalid AND instruction format" << std::endl;
            m_currentInstructionIndex++;
            return true;
        }
        int64_t src0 = getSourceValue(instr.sources[0]);
        int64_t src1 = getSourceValue(instr.sources[1]);
        int64_t result = src0 & src1;
        storeRegisterValue(instr.dest.registerIndex, static_cast<uint64_t>(result));
        m_currentInstructionIndex++;
        return true;
    }
    bool executeOR(const DecodedInstruction& instr) {
        if (instr.dest.type != OperandType::REGISTER || instr.sources.size() != 2) {
            std::cerr << "Invalid OR instruction format" << std::endl;
            m_currentInstructionIndex++;
            return true;
        }
        int64_t src0 = getSourceValue(instr.sources[0]);
        int64_t src1 = getSourceValue(instr.sources[1]);
        int64_t result = src0 | src1;
        storeRegisterValue(instr.dest.registerIndex, static_cast<uint64_t>(result));
        m_currentInstructionIndex++;
        return true;
    }
    bool executeXOR(const DecodedInstruction& instr) {
        if (instr.dest.type != OperandType::REGISTER || instr.sources.size() != 2) {
            std::cerr << "Invalid XOR instruction format" << std::endl;
            m_currentInstructionIndex++;
            return true;
        }
        int64_t src0 = getSourceValue(instr.sources[0]);
        int64_t src1 = getSourceValue(instr.sources[1]);
        int64_t result = src0 ^ src1;
        storeRegisterValue(instr.dest.registerIndex, static_cast<uint64_t>(result));
        m_currentInstructionIndex++;
        return true;
    }
    bool executeNOT(const DecodedInstruction& instr) {
        if (instr.dest.type != OperandType::REGISTER || instr.sources.size() != 1) {
            std::cerr << "Invalid NOT instruction format" << std::endl;
            m_currentInstructionIndex++;
            return true;
        }
        int64_t src = getSourceValue(instr.sources[0]);
        int64_t result = ~src;
        storeRegisterValue(instr.dest.registerIndex, static_cast<uint64_t>(result));
        m_currentInstructionIndex++;
        return true;
    }
    bool executeSHL(const DecodedInstruction& instr) {
        if (instr.dest.type != OperandType::REGISTER || instr.sources.size() != 2) {
            std::cerr << "Invalid SHL instruction format" << std::endl;
            m_currentInstructionIndex++;
            return true;
        }
        int64_t src0 = getSourceValue(instr.sources[0]);
        int64_t src1 = getSourceValue(instr.sources[1]);
        int64_t result = src0 << src1;
        storeRegisterValue(instr.dest.registerIndex, static_cast<uint64_t>(result));
        m_currentInstructionIndex++;
        return true;
    }
    bool executeSHR(const DecodedInstruction& instr) {
        if (instr.dest.type != OperandType::REGISTER || instr.sources.size() != 2) {
            std::cerr << "Invalid SHR instruction format" << std::endl;
            m_currentInstructionIndex++;
            return true;
        }
        int64_t src0 = getSourceValue(instr.sources[0]);
        int64_t src1 = getSourceValue(instr.sources[1]);
        int64_t result = src0 >> src1;
        storeRegisterValue(instr.dest.registerIndex, static_cast<uint64_t>(result));
        m_currentInstructionIndex++;
        return true;
    }
    bool executeNEG(const DecodedInstruction& instr) {
        if (instr.dest.type != OperandType::REGISTER || instr.sources.size() != 1) {
            std::cerr << "Invalid NEG instruction format" << std::endl;
            m_currentInstructionIndex++;
            return true;
        }
        int64_t src = getSourceValue(instr.sources[0]);
        int64_t result = -src;
        storeRegisterValue(instr.dest.registerIndex, static_cast<uint64_t>(result));
        m_currentInstructionIndex++;
        return true;
    }
    bool executeABS(const DecodedInstruction& instr) {
        if (instr.dest.type != OperandType::REGISTER || instr.sources.size() != 1) {
            std::cerr << "Invalid ABS instruction format" << std::endl;
            m_currentInstructionIndex++;
            return true;
        }
        int64_t src = getSourceValue(instr.sources[0]);
        int64_t result = src < 0 ? -src : src;
        storeRegisterValue(instr.dest.registerIndex, static_cast<uint64_t>(result));
        m_currentInstructionIndex++;
        return true;
    }
    bool executeJUMP(const DecodedInstruction& instr) {
        // 跳转到立即数或寄存器指定的指令索引
        if (instr.sources.size() != 1) {
            std::cerr << "Invalid JUMP instruction format" << std::endl;
            m_currentInstructionIndex++;
            return true;
        }
        size_t target = 0;
        if (instr.sources[0].type == OperandType::IMMEDIATE) {
            target = static_cast<size_t>(instr.sources[0].immediateValue);
        } else if (instr.sources[0].type == OperandType::REGISTER) {
            target = static_cast<size_t>(getSourceValue(instr.sources[0]));
        } else {
            std::cerr << "Unsupported JUMP target type" << std::endl;
            m_currentInstructionIndex++;
            return true;
        }
        m_currentInstructionIndex = target;
        return true;
    }
    bool executeCALL(const DecodedInstruction& instr) {
        // CALL instruction: call function_name, (arg1, arg2, ...)
        // For now, we'll use a simplified approach
        // In PTX, call format is: call (retval), function_name, (args);
        
        if (instr.sources.size() < 1) {
            std::cerr << "Invalid CALL instruction format" << std::endl;
            m_currentInstructionIndex++;
            return true;
        }
        
        // Get target address/label
        size_t target = 0;
        if (instr.sources[0].type == OperandType::IMMEDIATE) {
            target = static_cast<size_t>(instr.sources[0].immediateValue);
        } else if (instr.sources[0].type == OperandType::REGISTER) {
            target = static_cast<size_t>(getSourceValue(instr.sources[0]));
        } else {
            std::cerr << "Unsupported CALL target type" << std::endl;
            m_currentInstructionIndex++;
            return true;
        }
        
        // If we have program structure, try to resolve function by address
        if (m_hasProgramStructure) {
            // Find function that contains this target address
            for (const auto& func : m_program.functions) {
                if (target >= func.startInstructionIndex && target <= func.endInstructionIndex) {
                    // Found the function - use proper call mechanism
                    std::vector<uint64_t> args;
                    // TODO: Extract arguments from instruction or registers
                    return callFunction(func.name, args);
                }
            }
        }
        
        // Fallback: simple jump with return address save
        // Create a minimal call frame
        CallFrame frame;
        frame.functionName = "<unknown>";
        frame.returnAddress = m_currentInstructionIndex + 1;
        m_callStack.push_back(frame);
        
        m_currentInstructionIndex = target;
        return true;
    }
    bool executeCMOV(const DecodedInstruction& instr) {
        // 条件移动，假设第一个源为条件，第二个为值
        if (instr.dest.type != OperandType::REGISTER || instr.sources.size() != 2) {
            std::cerr << "Invalid CMOV instruction format" << std::endl;
            m_currentInstructionIndex++;
            return true;
        }
        int64_t cond = getSourceValue(instr.sources[0]);
        int64_t val = getSourceValue(instr.sources[1]);
        if (cond) {
            storeRegisterValue(instr.dest.registerIndex, static_cast<uint64_t>(val));
        }
        m_currentInstructionIndex++;
        return true;
    }
    bool executeSYNC(const DecodedInstruction& instr) {
        // 简单实现：同步点，实际应与warp调度/线程同步机制结合
        // 这里只是占位
        m_currentInstructionIndex++;
        return true;
    }
    bool executeMEMBAR(const DecodedInstruction& instr) {
        // 内存屏障，占位
        m_currentInstructionIndex++;
        return true;
    }
    bool executeBARRIER(const DecodedInstruction& instr) {
        // 屏障，占位
        m_currentInstructionIndex++;
        return true;
    }

    // Execute ADD instruction
    bool executeADD(const DecodedInstruction& instr) {
        if (instr.dest.type != OperandType::REGISTER || instr.sources.size() != 2) {
            std::cerr << "Invalid ADD instruction format" << std::endl;
            m_currentInstructionIndex++;
            return true;
        }
        
        // Get source operands
        int64_t src0 = getSourceValue(instr.sources[0]);
        int64_t src1 = getSourceValue(instr.sources[1]);
        
        // Perform addition
        int64_t result = src0 + src1;
        
        // Store result in destination register
        storeRegisterValue(instr.dest.registerIndex, static_cast<uint64_t>(result));
        
        // Move to next instruction
        m_currentInstructionIndex++;
        return true;
    }
    
    // Execute SUB instruction
    bool executeSUB(const DecodedInstruction& instr) {
        if (instr.dest.type != OperandType::REGISTER || instr.sources.size() != 2) {
            std::cerr << "Invalid SUB instruction format" << std::endl;
            m_currentInstructionIndex++;
            return true;
        }
        
        // Get source operands
        int64_t src0 = getSourceValue(instr.sources[0]);
        int64_t src1 = getSourceValue(instr.sources[1]);
        
        // Perform subtraction
        int64_t result = src0 - src1;
        
        // Store result in destination register
        storeRegisterValue(instr.dest.registerIndex, static_cast<uint64_t>(result));
        
        // Move to next instruction
        m_currentInstructionIndex++;
        return true;
    }
    
    // Execute MUL instruction
    bool executeMUL(const DecodedInstruction& instr) {
        if (instr.dest.type != OperandType::REGISTER || instr.sources.size() != 2) {
            std::cerr << "Invalid MUL instruction format" << std::endl;
            m_currentInstructionIndex++;
            return true;
        }
        
        // Get source operands
        int64_t src0 = getSourceValue(instr.sources[0]);
        int64_t src1 = getSourceValue(instr.sources[1]);
        
        // Perform multiplication
        int64_t result = src0 * src1;
        
        // Store result in destination register
        storeRegisterValue(instr.dest.registerIndex, static_cast<uint64_t>(result));
        
        // Move to next instruction
        m_currentInstructionIndex++;
        return true;
    }
    
    // Execute DIV instruction
    bool executeDIV(const DecodedInstruction& instr) {
        if (instr.dest.type != OperandType::REGISTER || instr.sources.size() != 2) {
            std::cerr << "Invalid DIV instruction format" << std::endl;
            m_currentInstructionIndex++;
            return true;
        }
        
        // Get source operands
        int64_t src0 = getSourceValue(instr.sources[0]);
        int64_t src1 = getSourceValue(instr.sources[1]);
        
        if (src1 == 0) {
            std::cerr << "Division by zero" << std::endl;
            m_currentInstructionIndex++;
            return true;
        }
        
        // Perform division
        int64_t result = src0 / src1;
        
        // Store result in destination register
        storeRegisterValue(instr.dest.registerIndex, static_cast<uint64_t>(result));
        
        // Move to next instruction
        m_currentInstructionIndex++;
        return true;
    }
    
    // Execute MOV instruction
    bool executeMOV(const DecodedInstruction& instr) {
        if (instr.dest.type != OperandType::REGISTER || instr.sources.size() != 1) {
            std::cerr << "Invalid MOV instruction format" << std::endl;
            m_currentInstructionIndex++;
            return true;
        }
        
        // Handle based on data type (mov.f32, mov.s32, etc.)
        if (instr.dataType == DataType::F32) {
            // Floating point move
            float src;
            if (instr.sources[0].type == OperandType::IMMEDIATE) {
                // Convert immediate to float
                uint32_t bits = static_cast<uint32_t>(instr.sources[0].immediateValue);
                std::memcpy(&src, &bits, sizeof(float));
            } else if (instr.sources[0].type == OperandType::REGISTER) {
                src = m_registerBank->readFloatRegister(instr.sources[0].registerIndex);
            } else {
                std::cerr << "Invalid MOV.F32 source operand type" << std::endl;
                m_currentInstructionIndex++;
                return true;
            }
            
            m_registerBank->writeFloatRegister(instr.dest.registerIndex, src);
        } else if (instr.dataType == DataType::F64) {
            // Double move
            double src;
            if (instr.sources[0].type == OperandType::IMMEDIATE) {
                uint64_t bits = instr.sources[0].immediateValue;
                std::memcpy(&src, &bits, sizeof(double));
            } else if (instr.sources[0].type == OperandType::REGISTER) {
                src = m_registerBank->readDoubleRegister(instr.sources[0].registerIndex);
            } else {
                std::cerr << "Invalid MOV.F64 source operand type" << std::endl;
                m_currentInstructionIndex++;
                return true;
            }
            
            m_registerBank->writeDoubleRegister(instr.dest.registerIndex, src);
        } else {
            // Integer move (default)
            int64_t src = getSourceValue(instr.sources[0]);
            m_registerBank->writeRegister(instr.dest.registerIndex, static_cast<uint64_t>(src));
        }
        
        // Move to next instruction
        m_currentInstructionIndex++;
        return true;
    }
    
    // Execute LD (load) instruction
    bool executeLD(const DecodedInstruction& instr) {
        if (instr.dest.type != OperandType::REGISTER || instr.sources.size() != 1 || 
            instr.sources[0].type != OperandType::MEMORY) {
            std::cerr << "Invalid LD instruction format" << std::endl;
            m_currentInstructionIndex++;
            return true;
        }
        
        // Calculate memory address
        uint64_t address = instr.sources[0].address;
        
        // Handle register-indirect addressing [%rX] or [%rX+offset]
        if (instr.sources[0].isIndirect) {
            // Get base address from register (use baseRegisterIndex for memory operands)
            uint64_t baseAddr = m_registerBank->readRegister(instr.sources[0].baseRegisterIndex);
            address = baseAddr + instr.sources[0].address;  // Add offset if any
        }
        
        // Use memory space from instruction (parsed from PTX modifiers)
        MemorySpace space = instr.memorySpace;
        
        // Increment appropriate memory read counter
        switch (space) {
            case MemorySpace::GLOBAL:
                m_performanceCounters->increment(PerformanceCounterIDs::GLOBAL_MEMORY_READS);
                break;
            case MemorySpace::SHARED:
                m_performanceCounters->increment(PerformanceCounterIDs::SHARED_MEMORY_READS);
                break;
            case MemorySpace::LOCAL:
                m_performanceCounters->increment(PerformanceCounterIDs::LOCAL_MEMORY_READS);
                break;
            case MemorySpace::PARAMETER:
                m_performanceCounters->increment(PerformanceCounterIDs::PARAMETER_MEMORY_READS);
                break;
            default:
                // Handle other memory spaces
                break;
        }
        
        // Read from memory based on data type
        uint64_t value = 0;
        switch (instr.dataType) {
            case DataType::S8:
                // Read as unsigned, then sign-extend
                value = static_cast<uint64_t>(static_cast<int64_t>(static_cast<int8_t>(m_memorySubsystem->read<uint8_t>(space, address))));
                break;
            case DataType::U8:
                value = static_cast<uint64_t>(m_memorySubsystem->read<uint8_t>(space, address));
                break;
            case DataType::S16:
                // Read as unsigned, then sign-extend
                value = static_cast<uint64_t>(static_cast<int64_t>(static_cast<int16_t>(m_memorySubsystem->read<uint16_t>(space, address))));
                break;
            case DataType::U16:
                value = static_cast<uint64_t>(m_memorySubsystem->read<uint16_t>(space, address));
                break;
            case DataType::S32:
                // Read as unsigned, then sign-extend
                value = static_cast<uint64_t>(static_cast<int64_t>(static_cast<int32_t>(m_memorySubsystem->read<uint32_t>(space, address))));
                break;
            case DataType::U32:
            case DataType::F32:
                value = static_cast<uint64_t>(m_memorySubsystem->read<uint32_t>(space, address));
                break;
            case DataType::S64:
            case DataType::U64:
            case DataType::F64:
            default:
                value = m_memorySubsystem->read<uint64_t>(space, address);
                break;
        }
        
        // Store result in destination register
        storeRegisterValue(instr.dest.registerIndex, value);
        
        // Move to next instruction
        m_currentInstructionIndex++;
        return true;
    }
    
    // Execute ST (store) instruction
    bool executeST(const DecodedInstruction& instr) {
        // New convention: dest is memory address, sources[0] is data value
        if (instr.dest.type != OperandType::MEMORY || instr.sources.size() != 1) {
            std::cerr << "Invalid ST instruction format" << std::endl;
            m_currentInstructionIndex++;
            return true;
        }
        
        // Get source value to store
        int64_t src = getSourceValue(instr.sources[0]);
        
        // Calculate memory address
        uint64_t address = instr.dest.address;
        
        // Handle register-indirect addressing [%rX] or [%rX+offset]
        if (instr.dest.isIndirect) {
            // Get base address from register (use baseRegisterIndex for memory operands)
            uint64_t baseAddr = m_registerBank->readRegister(instr.dest.baseRegisterIndex);
            std::cout << "ST DEBUG: baseRegisterIndex=" << instr.dest.baseRegisterIndex 
                      << ", baseAddr=0x" << std::hex << baseAddr 
                      << ", offset=0x" << instr.dest.address << std::dec << std::endl;
            address = baseAddr + instr.dest.address;  // Add offset if any
        }
        
        // Use memory space from instruction (parsed from PTX modifiers)
        MemorySpace space = instr.memorySpace;
        
        // Increment appropriate memory write counter
        switch (space) {
            case MemorySpace::GLOBAL:
                m_performanceCounters->increment(PerformanceCounterIDs::GLOBAL_MEMORY_WRITES);
                break;
            case MemorySpace::SHARED:
                m_performanceCounters->increment(PerformanceCounterIDs::SHARED_MEMORY_WRITES);
                break;
            case MemorySpace::LOCAL:
                m_performanceCounters->increment(PerformanceCounterIDs::LOCAL_MEMORY_WRITES);
                break;
            case MemorySpace::PARAMETER:
                m_performanceCounters->increment(PerformanceCounterIDs::PARAMETER_MEMORY_WRITES);
                break;
            default:
                // Handle other memory spaces
                break;
        }
        
        // DEBUG: Print store operation
        std::cout << "ST: writing value " << src << " to address 0x" << std::hex << address << std::dec 
                  << " in space " << static_cast<int>(space) 
                  << " dataType=" << static_cast<int>(instr.dataType) << std::endl;
        
        // Write to memory based on data type
        switch (instr.dataType) {
            case DataType::S8:
            case DataType::U8:
                m_memorySubsystem->write<uint8_t>(space, address, static_cast<uint8_t>(src));
                break;
            case DataType::S16:
            case DataType::U16:
                m_memorySubsystem->write<uint16_t>(space, address, static_cast<uint16_t>(src));
                break;
            case DataType::S32:
            case DataType::U32:
            case DataType::F32:
                m_memorySubsystem->write<uint32_t>(space, address, static_cast<uint32_t>(src));
                break;
            case DataType::S64:
            case DataType::U64:
            case DataType::F64:
            default:
                m_memorySubsystem->write<uint64_t>(space, address, static_cast<uint64_t>(src));
                break;
        }
        
        // Move to next instruction
        m_currentInstructionIndex++;
        return true;
    }
    
    // Execute BRA (branch) instruction
    bool executeBRA(const DecodedInstruction& instr) {
        // BRA should have exactly 1 source operand (the branch target)
        if (instr.sources.size() != 1) {
            std::cerr << "Invalid BRA instruction: expected 1 source operand, got " 
                      << instr.sources.size() << std::endl;
            m_currentInstructionIndex++;
            return true;
        }
        
        // Get branch target
        size_t target = m_currentInstructionIndex + 1; // Default: next instruction
        bool targetResolved = false;
        
        if (instr.sources[0].type == OperandType::LABEL) {
            // Branch to label - resolve label to instruction address
            if (resolveLabel(instr.sources[0].labelName, target)) {
                targetResolved = true;
            } else {
                std::cerr << "Failed to resolve label: " << instr.sources[0].labelName << std::endl;
                m_currentInstructionIndex++;
                return true;
            }
        } else if (instr.sources[0].type == OperandType::IMMEDIATE) {
            // Direct branch with immediate address
            target = static_cast<size_t>(instr.sources[0].immediateValue);
            targetResolved = true;
        } else if (instr.sources[0].type == OperandType::REGISTER) {
            // Indirect branch - address in register
            int64_t regValue = getSourceValue(instr.sources[0]);
            target = static_cast<size_t>(regValue);
            targetResolved = true;
        } else {
            std::cerr << "Unsupported branch target type: " << static_cast<int>(instr.sources[0].type) << std::endl;
            m_currentInstructionIndex++;
            return true;
        }
        
        // Increment branch counter
        m_performanceCounters->increment(PerformanceCounterIDs::BRANCHES);
        
        // Branch to target
        if (targetResolved) {
            if (target == m_currentInstructionIndex + 1) {
                // This is a sequential branch, not divergent
                m_currentInstructionIndex++;
            } else {
                // This is a non-sequential branch
                m_currentInstructionIndex = target;
                // Increment divergent branch counter
                m_performanceCounters->increment(PerformanceCounterIDs::DIVERGENT_BRANCHES);
            }
        }
        
        return true;
    }
    
    // Execute EXIT/RET instruction
    bool executeEXIT(const DecodedInstruction& instr) {
        // RET instruction returns from current function
        return returnFromFunction();
    }
    
    // Execute NOP instruction
    bool executeNOP(const DecodedInstruction& instr) {
        // Do nothing, just move to next instruction
        m_currentInstructionIndex++;
        return true;
    }
    
    // Execute LD_PARAM (load from parameter memory) instruction
    static bool executeLDParam(Impl& impl, const DecodedInstruction& instr) {
        if (instr.dest.type != OperandType::REGISTER || instr.sources.size() != 1) {
            std::cerr << "Invalid LD_PARAM instruction format" << std::endl;
            impl.m_currentInstructionIndex++;
            return true;
        }
        
        // Get the parameter offset or name
        uint64_t paramValue = 0;
        
        // Try to resolve parameter by name if we have program structure
        // For now, we use offset-based access
        uint64_t paramOffset = 0;
        if (instr.sources[0].type == OperandType::IMMEDIATE) {
            paramOffset = static_cast<uint64_t>(instr.sources[0].immediateValue);
        } else if (instr.sources[0].type == OperandType::REGISTER) {
            paramOffset = impl.m_registerBank->readRegister(instr.sources[0].registerIndex);
        } else if (instr.sources[0].type == OperandType::MEMORY) {
            // [param_name] - the address field contains the resolved parameter offset
            paramOffset = instr.sources[0].address;
        } else {
            std::cerr << "Invalid source operand type for LD_PARAM: type=" << static_cast<int>(instr.sources[0].type) << std::endl;
            impl.m_currentInstructionIndex++;
            return true;
        }
        
        // DEBUG: Print LD_PARAM operation
        std::cout << "LD_PARAM: loading from offset " << paramOffset 
                  << " (source type=" << static_cast<int>(instr.sources[0].type) << ")" << std::endl;
        
        // Read from parameter memory
        // Note: Use buffer-relative addressing (paramOffset directly), not absolute address
        // The PARAMETER memory space buffer starts at offset 0
        paramValue = impl.m_memorySubsystem->read<uint64_t>(MemorySpace::PARAMETER, paramOffset);
        
        // DEBUG: Print loaded value
        std::cout << "LD_PARAM: loaded value 0x" << std::hex << paramValue << std::dec 
                  << " into %r" << instr.dest.registerIndex << std::endl;
        
        // Store result in destination register
        impl.storeRegisterValue(instr.dest.registerIndex, paramValue);
        
        // Move to next instruction
        impl.m_currentInstructionIndex++;
        return true;
    }
    
    // Execute ST_PARAM (store to parameter memory) instruction
    static bool executeSTParam(Impl& impl, const DecodedInstruction& instr) {
        if (instr.sources.size() != 2) {
            std::cerr << "Invalid ST_PARAM instruction format" << std::endl;
            impl.m_currentInstructionIndex++;
            return true;
        }
        
        // Get the destination operand (parameter memory address)
        uint64_t paramOffset = 0;
        if (instr.sources[0].type == OperandType::IMMEDIATE) {
            paramOffset = static_cast<uint64_t>(instr.sources[0].immediateValue);
        } else if (instr.sources[0].type == OperandType::REGISTER) {
            paramOffset = impl.m_registerBank->readRegister(instr.sources[0].registerIndex);
        } else {
            std::cerr << "Invalid destination operand type for ST_PARAM" << std::endl;
            impl.m_currentInstructionIndex++;
            return true;
        }
        
        // Get source value
        uint64_t srcValue = 0;
        if (instr.sources[1].type == OperandType::IMMEDIATE) {
            srcValue = static_cast<uint64_t>(instr.sources[1].immediateValue);
        } else if (instr.sources[1].type == OperandType::REGISTER) {
            srcValue = impl.m_registerBank->readRegister(instr.sources[1].registerIndex);
        } else {
            std::cerr << "Invalid source operand type for ST_PARAM" << std::endl;
            impl.m_currentInstructionIndex++;
            return true;
        }
        
        // Write to parameter memory
        // Note: Use buffer-relative addressing (paramOffset directly), not absolute address
        // The PARAMETER memory space buffer starts at offset 0
        impl.m_memorySubsystem->write<uint64_t>(MemorySpace::PARAMETER, paramOffset, srcValue);
        
        // Move to next instruction
        impl.m_currentInstructionIndex++;
        return true;
    }
    
    // ===== Floating-Point Instruction Execution =====
    
    // Execute ADD.F32 instruction
    bool executeADD_F32(const DecodedInstruction& instr) {
        if (instr.dest.type != OperandType::REGISTER || instr.sources.size() != 2) {
            std::cerr << "Invalid ADD.F32 instruction format" << std::endl;
            m_currentInstructionIndex++;
            return true;
        }
        
        // Read source operands (float registers)
        float src1 = m_registerBank->readFloatRegister(instr.sources[0].registerIndex);
        float src2 = (instr.sources[1].type == OperandType::IMMEDIATE) 
                     ? *reinterpret_cast<const float*>(&instr.sources[1].immediateValue)
                     : m_registerBank->readFloatRegister(instr.sources[1].registerIndex);
        
        // Perform floating-point addition
        float result = src1 + src2;
        
        // Write back to destination register
        m_registerBank->writeFloatRegister(instr.dest.registerIndex, result);
        
        // Update performance counters
        m_performanceCounters->increment(PerformanceCounterIDs::INSTRUCTIONS_EXECUTED);
        
        m_currentInstructionIndex++;
        return true;
    }
    
    // Execute SUB.F32 instruction
    bool executeSUB_F32(const DecodedInstruction& instr) {
        if (instr.dest.type != OperandType::REGISTER || instr.sources.size() != 2) {
            std::cerr << "Invalid SUB.F32 instruction format" << std::endl;
            m_currentInstructionIndex++;
            return true;
        }
        
        float src1 = m_registerBank->readFloatRegister(instr.sources[0].registerIndex);
        float src2 = (instr.sources[1].type == OperandType::IMMEDIATE)
                     ? *reinterpret_cast<const float*>(&instr.sources[1].immediateValue)
                     : m_registerBank->readFloatRegister(instr.sources[1].registerIndex);
        
        float result = src1 - src2;
        m_registerBank->writeFloatRegister(instr.dest.registerIndex, result);
        m_performanceCounters->increment(PerformanceCounterIDs::INSTRUCTIONS_EXECUTED);
        
        m_currentInstructionIndex++;
        return true;
    }
    
    // Execute MUL.F32 instruction
    bool executeMUL_F32(const DecodedInstruction& instr) {
        if (instr.dest.type != OperandType::REGISTER || instr.sources.size() != 2) {
            std::cerr << "Invalid MUL.F32 instruction format" << std::endl;
            m_currentInstructionIndex++;
            return true;
        }
        
        float src1 = m_registerBank->readFloatRegister(instr.sources[0].registerIndex);
        float src2 = (instr.sources[1].type == OperandType::IMMEDIATE)
                     ? *reinterpret_cast<const float*>(&instr.sources[1].immediateValue)
                     : m_registerBank->readFloatRegister(instr.sources[1].registerIndex);
        
        float result = src1 * src2;
        m_registerBank->writeFloatRegister(instr.dest.registerIndex, result);
        m_performanceCounters->increment(PerformanceCounterIDs::INSTRUCTIONS_EXECUTED);
        
        m_currentInstructionIndex++;
        return true;
    }
    
    // Execute DIV.F32 instruction
    bool executeDIV_F32(const DecodedInstruction& instr) {
        if (instr.dest.type != OperandType::REGISTER || instr.sources.size() != 2) {
            std::cerr << "Invalid DIV.F32 instruction format" << std::endl;
            m_currentInstructionIndex++;
            return true;
        }
        
        float src1 = m_registerBank->readFloatRegister(instr.sources[0].registerIndex);
        float src2 = (instr.sources[1].type == OperandType::IMMEDIATE)
                     ? *reinterpret_cast<const float*>(&instr.sources[1].immediateValue)
                     : m_registerBank->readFloatRegister(instr.sources[1].registerIndex);
        
        if (src2 == 0.0f) {
            std::cerr << "Division by zero in DIV.F32" << std::endl;
            m_currentInstructionIndex++;
            return true;
        }
        
        float result = src1 / src2;
        m_registerBank->writeFloatRegister(instr.dest.registerIndex, result);
        m_performanceCounters->increment(PerformanceCounterIDs::INSTRUCTIONS_EXECUTED);
        
        m_currentInstructionIndex++;
        return true;
    }
    
    // Execute FMA.F32 instruction (fused multiply-add)
    bool executeFMA_F32(const DecodedInstruction& instr) {
        if (instr.dest.type != OperandType::REGISTER || instr.sources.size() != 3) {
            std::cerr << "Invalid FMA.F32 instruction format" << std::endl;
            m_currentInstructionIndex++;
            return true;
        }
        
        // fma.f32 %f0, %f1, %f2, %f3;  // %f0 = %f1 * %f2 + %f3
        float src1 = m_registerBank->readFloatRegister(instr.sources[0].registerIndex);
        float src2 = m_registerBank->readFloatRegister(instr.sources[1].registerIndex);
        float src3 = m_registerBank->readFloatRegister(instr.sources[2].registerIndex);
        
        // Use FMA if available, otherwise simulate
        float result = src1 * src2 + src3;
        
        m_registerBank->writeFloatRegister(instr.dest.registerIndex, result);
        m_performanceCounters->increment(PerformanceCounterIDs::INSTRUCTIONS_EXECUTED);
        
        m_currentInstructionIndex++;
        return true;
    }
    
    // Execute SQRT.F32 instruction
    bool executeSQRT_F32(const DecodedInstruction& instr) {
        if (instr.dest.type != OperandType::REGISTER || instr.sources.size() != 1) {
            std::cerr << "Invalid SQRT.F32 instruction format" << std::endl;
            m_currentInstructionIndex++;
            return true;
        }
        
        float src = m_registerBank->readFloatRegister(instr.sources[0].registerIndex);
        float result = std::sqrt(src);
        m_registerBank->writeFloatRegister(instr.dest.registerIndex, result);
        m_performanceCounters->increment(PerformanceCounterIDs::INSTRUCTIONS_EXECUTED);
        
        m_currentInstructionIndex++;
        return true;
    }
    
    // Execute NEG.F32 instruction
    bool executeNEG_F32(const DecodedInstruction& instr) {
        if (instr.dest.type != OperandType::REGISTER || instr.sources.size() != 1) {
            std::cerr << "Invalid NEG.F32 instruction format" << std::endl;
            m_currentInstructionIndex++;
            return true;
        }
        
        float src = m_registerBank->readFloatRegister(instr.sources[0].registerIndex);
        float result = -src;
        m_registerBank->writeFloatRegister(instr.dest.registerIndex, result);
        m_performanceCounters->increment(PerformanceCounterIDs::INSTRUCTIONS_EXECUTED);
        
        m_currentInstructionIndex++;
        return true;
    }
    
    // Execute ABS.F32 instruction
    bool executeABS_F32(const DecodedInstruction& instr) {
        if (instr.dest.type != OperandType::REGISTER || instr.sources.size() != 1) {
            std::cerr << "Invalid ABS.F32 instruction format" << std::endl;
            m_currentInstructionIndex++;
            return true;
        }
        
        float src = m_registerBank->readFloatRegister(instr.sources[0].registerIndex);
        float result = std::abs(src);
        m_registerBank->writeFloatRegister(instr.dest.registerIndex, result);
        m_performanceCounters->increment(PerformanceCounterIDs::INSTRUCTIONS_EXECUTED);
        
        m_currentInstructionIndex++;
        return true;
    }
    
    // ===== Comparison Instructions =====
    
    // Execute SETP instruction
    bool executeSETP(const DecodedInstruction& instr) {
        // SETP can write to either predicate registers (%p) or integer registers (%r)
        if ((instr.dest.type != OperandType::PREDICATE && instr.dest.type != OperandType::REGISTER) 
            || instr.sources.size() != 2) {
            std::cerr << "Invalid SETP instruction format" << std::endl;
            m_currentInstructionIndex++;
            return true;
        }
        
        bool result = false;
        
        // Get source values using getSourceValue (handles both registers and immediates)
        int64_t src1_val = getSourceValue(instr.sources[0]);
        int64_t src2_val = getSourceValue(instr.sources[1]);
        
        // Compare based on data type
        if (instr.dataType == DataType::S32) {
            int32_t src1 = static_cast<int32_t>(src1_val);
            int32_t src2 = static_cast<int32_t>(src2_val);
            
            switch (instr.compareOp) {
                case CompareOp::LT: result = (src1 < src2); break;
                case CompareOp::LE: result = (src1 <= src2); break;
                case CompareOp::GT: result = (src1 > src2); break;
                case CompareOp::GE: result = (src1 >= src2); break;
                case CompareOp::EQ: result = (src1 == src2); break;
                case CompareOp::NE: result = (src1 != src2); break;
                default: break;
            }
        } else if (instr.dataType == DataType::U32) {
            uint32_t src1 = static_cast<uint32_t>(src1_val);
            uint32_t src2 = static_cast<uint32_t>(src2_val);
            
            switch (instr.compareOp) {
                case CompareOp::LO: result = (src1 < src2); break;
                case CompareOp::LS: result = (src1 <= src2); break;
                case CompareOp::HI: result = (src1 > src2); break;
                case CompareOp::HS: result = (src1 >= src2); break;
                case CompareOp::EQ: result = (src1 == src2); break;
                case CompareOp::NE: result = (src1 != src2); break;
                default: break;
            }
        } else if (instr.dataType == DataType::F32) {
            // For float comparisons, interpret the int64_t as float bits
            float src1 = *reinterpret_cast<float*>(&src1_val);
            float src2 = *reinterpret_cast<float*>(&src2_val);
            
            switch (instr.compareOp) {
                case CompareOp::LT: result = (src1 < src2); break;
                case CompareOp::LE: result = (src1 <= src2); break;
                case CompareOp::GT: result = (src1 > src2); break;
                case CompareOp::GE: result = (src1 >= src2); break;
                case CompareOp::EQ: result = (src1 == src2); break;
                case CompareOp::NE: result = (src1 != src2); break;
                default: break;
            }
        } else if (instr.dataType == DataType::F64) {
            // For double comparisons, interpret the int64_t as double bits
            double src1 = *reinterpret_cast<double*>(&src1_val);
            double src2 = *reinterpret_cast<double*>(&src2_val);
            
            switch (instr.compareOp) {
                case CompareOp::LT: result = (src1 < src2); break;
                case CompareOp::LE: result = (src1 <= src2); break;
                case CompareOp::GT: result = (src1 > src2); break;
                case CompareOp::GE: result = (src1 >= src2); break;
                case CompareOp::EQ: result = (src1 == src2); break;
                case CompareOp::NE: result = (src1 != src2); break;
                default: break;
            }
        }
        
        // Write result - can be to predicate register or integer register
        if (instr.dest.type == OperandType::PREDICATE) {
            m_registerBank->writePredicate(instr.dest.predicateIndex, result);
        } else if (instr.dest.type == OperandType::REGISTER) {
            // Write to integer register (0 or 1)
            m_registerBank->writeRegister(instr.dest.registerIndex, result ? 1 : 0);
        }
        
        m_performanceCounters->increment(PerformanceCounterIDs::INSTRUCTIONS_EXECUTED);
        
        m_currentInstructionIndex++;
        return true;
    }
    
    // Execute SELP (conditional select) instruction
    bool executeSELP(const DecodedInstruction& instr) {
        if (instr.dest.type != OperandType::REGISTER || instr.sources.size() != 3) {
            std::cerr << "Invalid SELP instruction format" << std::endl;
            m_currentInstructionIndex++;
            return true;
        }
        
        // selp.s32 %r3, %r1, %r2, %p1;  // %r3 = %p1 ? %r1 : %r2
        
        // Read predicate (third source)
        bool pred = false;
        if (instr.sources[2].type == OperandType::PREDICATE) {
            pred = m_registerBank->readPredicate(instr.sources[2].predicateIndex);
        } else {
            std::cerr << "Invalid predicate operand in SELP" << std::endl;
            m_currentInstructionIndex++;
            return true;
        }
        
        // Select based on data type
        if (instr.dataType == DataType::S32 || instr.dataType == DataType::U32 || 
            instr.dataType == DataType::S64 || instr.dataType == DataType::U64) {
            // Integer types
            uint64_t src1 = m_registerBank->readRegister(instr.sources[0].registerIndex);
            uint64_t src2 = m_registerBank->readRegister(instr.sources[1].registerIndex);
            uint64_t result = pred ? src1 : src2;
            m_registerBank->writeRegister(instr.dest.registerIndex, result);
        } else if (instr.dataType == DataType::F32) {
            // Single precision float
            float src1 = m_registerBank->readFloatRegister(instr.sources[0].registerIndex);
            float src2 = m_registerBank->readFloatRegister(instr.sources[1].registerIndex);
            float result = pred ? src1 : src2;
            m_registerBank->writeFloatRegister(instr.dest.registerIndex, result);
        } else if (instr.dataType == DataType::F64) {
            // Double precision float
            double src1 = m_registerBank->readDoubleRegister(instr.sources[0].registerIndex);
            double src2 = m_registerBank->readDoubleRegister(instr.sources[1].registerIndex);
            double result = pred ? src1 : src2;
            m_registerBank->writeDoubleRegister(instr.dest.registerIndex, result);
        }
        
        m_performanceCounters->increment(PerformanceCounterIDs::INSTRUCTIONS_EXECUTED);
        m_currentInstructionIndex++;
        return true;
    }
    
    // Execute CVT (type conversion) instruction
    bool executeCVT(const DecodedInstruction& instr) {
        if (instr.dest.type != OperandType::REGISTER || instr.sources.size() != 1) {
            std::cerr << "Invalid CVT instruction format" << std::endl;
            m_currentInstructionIndex++;
            return true;
        }
        
        // cvt.dstType.srcType %dest, %src
        
        // Float to signed integer conversions
        if (instr.srcType == DataType::F32 && instr.dstType == DataType::S32) {
            float src = m_registerBank->readFloatRegister(instr.sources[0].registerIndex);
            int32_t dst = static_cast<int32_t>(src);
            m_registerBank->writeRegister(instr.dest.registerIndex, static_cast<uint64_t>(static_cast<int64_t>(dst)));
        }
        else if (instr.srcType == DataType::F64 && instr.dstType == DataType::S32) {
            double src = m_registerBank->readDoubleRegister(instr.sources[0].registerIndex);
            int32_t dst = static_cast<int32_t>(src);
            m_registerBank->writeRegister(instr.dest.registerIndex, static_cast<uint64_t>(static_cast<int64_t>(dst)));
        }
        else if (instr.srcType == DataType::F32 && instr.dstType == DataType::S64) {
            float src = m_registerBank->readFloatRegister(instr.sources[0].registerIndex);
            int64_t dst = static_cast<int64_t>(src);
            m_registerBank->writeRegister(instr.dest.registerIndex, static_cast<uint64_t>(dst));
        }
        else if (instr.srcType == DataType::F64 && instr.dstType == DataType::S64) {
            double src = m_registerBank->readDoubleRegister(instr.sources[0].registerIndex);
            int64_t dst = static_cast<int64_t>(src);
            m_registerBank->writeRegister(instr.dest.registerIndex, static_cast<uint64_t>(dst));
        }
        
        // Float to unsigned integer conversions
        else if (instr.srcType == DataType::F32 && instr.dstType == DataType::U32) {
            float src = m_registerBank->readFloatRegister(instr.sources[0].registerIndex);
            uint32_t dst = static_cast<uint32_t>(src);
            m_registerBank->writeRegister(instr.dest.registerIndex, static_cast<uint64_t>(dst));
        }
        else if (instr.srcType == DataType::F64 && instr.dstType == DataType::U32) {
            double src = m_registerBank->readDoubleRegister(instr.sources[0].registerIndex);
            uint32_t dst = static_cast<uint32_t>(src);
            m_registerBank->writeRegister(instr.dest.registerIndex, static_cast<uint64_t>(dst));
        }
        else if (instr.srcType == DataType::F32 && instr.dstType == DataType::U64) {
            float src = m_registerBank->readFloatRegister(instr.sources[0].registerIndex);
            uint64_t dst = static_cast<uint64_t>(src);
            m_registerBank->writeRegister(instr.dest.registerIndex, dst);
        }
        else if (instr.srcType == DataType::F64 && instr.dstType == DataType::U64) {
            double src = m_registerBank->readDoubleRegister(instr.sources[0].registerIndex);
            uint64_t dst = static_cast<uint64_t>(src);
            m_registerBank->writeRegister(instr.dest.registerIndex, dst);
        }
        
        // Signed integer to float conversions
        else if (instr.srcType == DataType::S32 && instr.dstType == DataType::F32) {
            int32_t src = static_cast<int32_t>(m_registerBank->readRegister(instr.sources[0].registerIndex));
            float dst = static_cast<float>(src);
            m_registerBank->writeFloatRegister(instr.dest.registerIndex, dst);
        }
        else if (instr.srcType == DataType::S64 && instr.dstType == DataType::F32) {
            int64_t src = static_cast<int64_t>(m_registerBank->readRegister(instr.sources[0].registerIndex));
            float dst = static_cast<float>(src);
            m_registerBank->writeFloatRegister(instr.dest.registerIndex, dst);
        }
        else if (instr.srcType == DataType::S32 && instr.dstType == DataType::F64) {
            int32_t src = static_cast<int32_t>(m_registerBank->readRegister(instr.sources[0].registerIndex));
            double dst = static_cast<double>(src);
            m_registerBank->writeDoubleRegister(instr.dest.registerIndex, dst);
        }
        else if (instr.srcType == DataType::S64 && instr.dstType == DataType::F64) {
            int64_t src = static_cast<int64_t>(m_registerBank->readRegister(instr.sources[0].registerIndex));
            double dst = static_cast<double>(src);
            m_registerBank->writeDoubleRegister(instr.dest.registerIndex, dst);
        }
        
        // Unsigned integer to float conversions
        else if (instr.srcType == DataType::U32 && instr.dstType == DataType::F32) {
            uint32_t src = static_cast<uint32_t>(m_registerBank->readRegister(instr.sources[0].registerIndex));
            float dst = static_cast<float>(src);
            m_registerBank->writeFloatRegister(instr.dest.registerIndex, dst);
        }
        else if (instr.srcType == DataType::U64 && instr.dstType == DataType::F32) {
            uint64_t src = m_registerBank->readRegister(instr.sources[0].registerIndex);
            float dst = static_cast<float>(src);
            m_registerBank->writeFloatRegister(instr.dest.registerIndex, dst);
        }
        else if (instr.srcType == DataType::U32 && instr.dstType == DataType::F64) {
            uint32_t src = static_cast<uint32_t>(m_registerBank->readRegister(instr.sources[0].registerIndex));
            double dst = static_cast<double>(src);
            m_registerBank->writeDoubleRegister(instr.dest.registerIndex, dst);
        }
        else if (instr.srcType == DataType::U64 && instr.dstType == DataType::F64) {
            uint64_t src = m_registerBank->readRegister(instr.sources[0].registerIndex);
            double dst = static_cast<double>(src);
            m_registerBank->writeDoubleRegister(instr.dest.registerIndex, dst);
        }
        
        // Float to float conversions (precision change)
        else if (instr.srcType == DataType::F32 && instr.dstType == DataType::F64) {
            float src = m_registerBank->readFloatRegister(instr.sources[0].registerIndex);
            double dst = static_cast<double>(src);
            m_registerBank->writeDoubleRegister(instr.dest.registerIndex, dst);
        }
        else if (instr.srcType == DataType::F64 && instr.dstType == DataType::F32) {
            double src = m_registerBank->readDoubleRegister(instr.sources[0].registerIndex);
            float dst = static_cast<float>(src);
            m_registerBank->writeFloatRegister(instr.dest.registerIndex, dst);
        }
        
        // Integer to integer conversions (size/sign change)
        else if (instr.srcType == DataType::S32 && instr.dstType == DataType::S64) {
            int32_t src = static_cast<int32_t>(m_registerBank->readRegister(instr.sources[0].registerIndex));
            int64_t dst = static_cast<int64_t>(src);
            m_registerBank->writeRegister(instr.dest.registerIndex, static_cast<uint64_t>(dst));
        }
        else if (instr.srcType == DataType::U32 && instr.dstType == DataType::U64) {
            uint32_t src = static_cast<uint32_t>(m_registerBank->readRegister(instr.sources[0].registerIndex));
            uint64_t dst = static_cast<uint64_t>(src);
            m_registerBank->writeRegister(instr.dest.registerIndex, dst);
        }
        else if (instr.srcType == DataType::S64 && instr.dstType == DataType::S32) {
            int64_t src = static_cast<int64_t>(m_registerBank->readRegister(instr.sources[0].registerIndex));
            int32_t dst = static_cast<int32_t>(src);
            m_registerBank->writeRegister(instr.dest.registerIndex, static_cast<uint64_t>(static_cast<int64_t>(dst)));
        }
        else if (instr.srcType == DataType::U64 && instr.dstType == DataType::U32) {
            uint64_t src = m_registerBank->readRegister(instr.sources[0].registerIndex);
            uint32_t dst = static_cast<uint32_t>(src);
            m_registerBank->writeRegister(instr.dest.registerIndex, static_cast<uint64_t>(dst));
        }
        
        else {
            std::cerr << "Unsupported CVT conversion: srcType=" << static_cast<int>(instr.srcType) 
                      << " dstType=" << static_cast<int>(instr.dstType) << std::endl;
        }
        
        m_performanceCounters->increment(PerformanceCounterIDs::INSTRUCTIONS_EXECUTED);
        m_currentInstructionIndex++;
        return true;
    }
    
    // ===== Atomic Operation Instructions =====
    
    // Execute ATOM.ADD (atomic add) instruction
    bool executeATOM_ADD(const DecodedInstruction& instr) {
        if (instr.sources.size() < 2) {
            std::cerr << "Invalid ATOM.ADD instruction format" << std::endl;
            m_currentInstructionIndex++;
            return true;
        }
        
        // atom.global.add.u32 %r1, [%rd1], %r2;
        // %r1 = old value at [%rd1]
        // [%rd1] = [%rd1] + %r2
        
        // Get memory address (first source)
        uint64_t address = 0;
        if (instr.sources[0].type == OperandType::MEMORY) {
            address = instr.sources[0].address;
        } else if (instr.sources[0].type == OperandType::REGISTER) {
            address = m_registerBank->readRegister(instr.sources[0].registerIndex);
        } else {
            std::cerr << "Invalid address operand in ATOM.ADD" << std::endl;
            m_currentInstructionIndex++;
            return true;
        }
        
        // Get value to add (second source)
        uint32_t addValue = static_cast<uint32_t>(
            m_registerBank->readRegister(instr.sources[1].registerIndex));
        
        // Determine memory space (default to GLOBAL)
        MemorySpace space = (instr.memorySpace != MemorySpace::GLOBAL && 
                            instr.memorySpace != MemorySpace::SHARED) 
                            ? MemorySpace::GLOBAL : instr.memorySpace;
        
        // 🔒 Atomic operation: read-modify-write
        // Note: In a real multi-threaded environment, this would need a mutex
        uint32_t oldValue = m_memorySubsystem->read<uint32_t>(space, address);
        uint32_t newValue = oldValue + addValue;
        m_memorySubsystem->write<uint32_t>(space, address, newValue);
        
        // Return old value to destination register
        if (instr.dest.type == OperandType::REGISTER) {
            m_registerBank->writeRegister(instr.dest.registerIndex, 
                                          static_cast<uint64_t>(oldValue));
        }
        
        m_performanceCounters->increment(PerformanceCounterIDs::INSTRUCTIONS_EXECUTED);
        m_currentInstructionIndex++;
        return true;
    }
    
    // Execute ATOM.SUB (atomic subtract) instruction
    bool executeATOM_SUB(const DecodedInstruction& instr) {
        if (instr.sources.size() < 2) {
            std::cerr << "Invalid ATOM.SUB instruction format" << std::endl;
            m_currentInstructionIndex++;
            return true;
        }
        
        uint64_t address = 0;
        if (instr.sources[0].type == OperandType::MEMORY) {
            address = instr.sources[0].address;
        } else if (instr.sources[0].type == OperandType::REGISTER) {
            address = m_registerBank->readRegister(instr.sources[0].registerIndex);
        }
        
        uint32_t subValue = static_cast<uint32_t>(
            m_registerBank->readRegister(instr.sources[1].registerIndex));
        
        MemorySpace space = (instr.memorySpace != MemorySpace::GLOBAL && 
                            instr.memorySpace != MemorySpace::SHARED) 
                            ? MemorySpace::GLOBAL : instr.memorySpace;
        
        uint32_t oldValue = m_memorySubsystem->read<uint32_t>(space, address);
        uint32_t newValue = oldValue - subValue;
        m_memorySubsystem->write<uint32_t>(space, address, newValue);
        
        if (instr.dest.type == OperandType::REGISTER) {
            m_registerBank->writeRegister(instr.dest.registerIndex, 
                                          static_cast<uint64_t>(oldValue));
        }
        
        m_performanceCounters->increment(PerformanceCounterIDs::INSTRUCTIONS_EXECUTED);
        m_currentInstructionIndex++;
        return true;
    }
    
    // Execute ATOM.EXCH (atomic exchange) instruction
    bool executeATOM_EXCH(const DecodedInstruction& instr) {
        if (instr.sources.size() < 2) {
            std::cerr << "Invalid ATOM.EXCH instruction format" << std::endl;
            m_currentInstructionIndex++;
            return true;
        }
        
        // atom.global.exch.b32 %r1, [%rd1], %r2;
        // %r1 = old value at [%rd1]
        // [%rd1] = %r2
        
        uint64_t address = 0;
        if (instr.sources[0].type == OperandType::MEMORY) {
            address = instr.sources[0].address;
        } else if (instr.sources[0].type == OperandType::REGISTER) {
            address = m_registerBank->readRegister(instr.sources[0].registerIndex);
        }
        
        uint32_t newValue = static_cast<uint32_t>(
            m_registerBank->readRegister(instr.sources[1].registerIndex));
        
        MemorySpace space = (instr.memorySpace != MemorySpace::GLOBAL && 
                            instr.memorySpace != MemorySpace::SHARED) 
                            ? MemorySpace::GLOBAL : instr.memorySpace;
        
        uint32_t oldValue = m_memorySubsystem->read<uint32_t>(space, address);
        m_memorySubsystem->write<uint32_t>(space, address, newValue);
        
        if (instr.dest.type == OperandType::REGISTER) {
            m_registerBank->writeRegister(instr.dest.registerIndex, 
                                          static_cast<uint64_t>(oldValue));
        }
        
        m_performanceCounters->increment(PerformanceCounterIDs::INSTRUCTIONS_EXECUTED);
        m_currentInstructionIndex++;
        return true;
    }
    
    // Execute ATOM.CAS (atomic compare-and-swap) instruction
    bool executeATOM_CAS(const DecodedInstruction& instr) {
        if (instr.sources.size() < 3) {
            std::cerr << "Invalid ATOM.CAS instruction format" << std::endl;
            m_currentInstructionIndex++;
            return true;
        }
        
        // atom.global.cas.b32 %r1, [%rd1], %r2, %r3;
        // if ([%rd1] == %r2) [%rd1] = %r3;
        // %r1 = old value at [%rd1]
        
        uint64_t address = 0;
        if (instr.sources[0].type == OperandType::MEMORY) {
            address = instr.sources[0].address;
        } else if (instr.sources[0].type == OperandType::REGISTER) {
            address = m_registerBank->readRegister(instr.sources[0].registerIndex);
        }
        
        uint32_t compareValue = static_cast<uint32_t>(
            m_registerBank->readRegister(instr.sources[1].registerIndex));
        uint32_t newValue = static_cast<uint32_t>(
            m_registerBank->readRegister(instr.sources[2].registerIndex));
        
        MemorySpace space = (instr.memorySpace != MemorySpace::GLOBAL && 
                            instr.memorySpace != MemorySpace::SHARED) 
                            ? MemorySpace::GLOBAL : instr.memorySpace;
        
        uint32_t oldValue = m_memorySubsystem->read<uint32_t>(space, address);
        
        // Only write if old value equals compare value
        if (oldValue == compareValue) {
            m_memorySubsystem->write<uint32_t>(space, address, newValue);
        }
        
        // Always return old value
        if (instr.dest.type == OperandType::REGISTER) {
            m_registerBank->writeRegister(instr.dest.registerIndex, 
                                          static_cast<uint64_t>(oldValue));
        }
        
        m_performanceCounters->increment(PerformanceCounterIDs::INSTRUCTIONS_EXECUTED);
        m_currentInstructionIndex++;
        return true;
    }
    
    // Execute ATOM.MIN (atomic minimum) instruction
    bool executeATOM_MIN(const DecodedInstruction& instr) {
        if (instr.sources.size() < 2) {
            std::cerr << "Invalid ATOM.MIN instruction format" << std::endl;
            m_currentInstructionIndex++;
            return true;
        }
        
        uint64_t address = 0;
        if (instr.sources[0].type == OperandType::MEMORY) {
            address = instr.sources[0].address;
        } else if (instr.sources[0].type == OperandType::REGISTER) {
            address = m_registerBank->readRegister(instr.sources[0].registerIndex);
        }
        
        uint32_t compareValue = static_cast<uint32_t>(
            m_registerBank->readRegister(instr.sources[1].registerIndex));
        
        MemorySpace space = (instr.memorySpace != MemorySpace::GLOBAL && 
                            instr.memorySpace != MemorySpace::SHARED) 
                            ? MemorySpace::GLOBAL : instr.memorySpace;
        
        uint32_t oldValue = m_memorySubsystem->read<uint32_t>(space, address);
        uint32_t newValue = (oldValue < compareValue) ? oldValue : compareValue;
        m_memorySubsystem->write<uint32_t>(space, address, newValue);
        
        if (instr.dest.type == OperandType::REGISTER) {
            m_registerBank->writeRegister(instr.dest.registerIndex, 
                                          static_cast<uint64_t>(oldValue));
        }
        
        m_performanceCounters->increment(PerformanceCounterIDs::INSTRUCTIONS_EXECUTED);
        m_currentInstructionIndex++;
        return true;
    }
    
    // Execute ATOM.MAX (atomic maximum) instruction
    bool executeATOM_MAX(const DecodedInstruction& instr) {
        if (instr.sources.size() < 2) {
            std::cerr << "Invalid ATOM.MAX instruction format" << std::endl;
            m_currentInstructionIndex++;
            return true;
        }
        
        uint64_t address = 0;
        if (instr.sources[0].type == OperandType::MEMORY) {
            address = instr.sources[0].address;
        } else if (instr.sources[0].type == OperandType::REGISTER) {
            address = m_registerBank->readRegister(instr.sources[0].registerIndex);
        }
        
        uint32_t compareValue = static_cast<uint32_t>(
            m_registerBank->readRegister(instr.sources[1].registerIndex));
        
        MemorySpace space = (instr.memorySpace != MemorySpace::GLOBAL && 
                            instr.memorySpace != MemorySpace::SHARED) 
                            ? MemorySpace::GLOBAL : instr.memorySpace;
        
        uint32_t oldValue = m_memorySubsystem->read<uint32_t>(space, address);
        uint32_t newValue = (oldValue > compareValue) ? oldValue : compareValue;
        m_memorySubsystem->write<uint32_t>(space, address, newValue);
        
        if (instr.dest.type == OperandType::REGISTER) {
            m_registerBank->writeRegister(instr.dest.registerIndex, 
                                          static_cast<uint64_t>(oldValue));
        }
        
        m_performanceCounters->increment(PerformanceCounterIDs::INSTRUCTIONS_EXECUTED);
        m_currentInstructionIndex++;
        return true;
    }
    
    // Helper function to get source operand value with memory access tracking
    int64_t getSourceValue(const Operand& operand) {
        switch (operand.type) {
            case OperandType::REGISTER:
                // Increment register read counter
                m_performanceCounters->increment(PerformanceCounterIDs::REGISTER_READS);
                return static_cast<int64_t>(m_registerBank->readRegister(operand.registerIndex));
            
            case OperandType::IMMEDIATE:
                return operand.immediateValue;
            
            case OperandType::MEMORY:
                {
                // Determine memory space
                MemorySpace space = determineMemorySpace(operand.address);
                
                // Increment appropriate memory read counter
                switch (space) {
                    case MemorySpace::GLOBAL:
                        m_performanceCounters->increment(PerformanceCounterIDs::GLOBAL_MEMORY_READS);
                        break;
                    case MemorySpace::SHARED:
                        m_performanceCounters->increment(PerformanceCounterIDs::SHARED_MEMORY_READS);
                        break;
                    case MemorySpace::PARAMETER:
                        m_performanceCounters->increment(PerformanceCounterIDs::PARAMETER_MEMORY_READS);
                        return static_cast<int64_t>(m_memorySubsystem->read<uint64_t>(space, operand.address));
                    
                    case MemorySpace::LOCAL:
                        m_performanceCounters->increment(PerformanceCounterIDs::LOCAL_MEMORY_READS);
                        return static_cast<int64_t>(m_memorySubsystem->read<uint64_t>(space, operand.address));
                    
                    default:
                        std::cerr << "Unknown operand type" << std::endl;
                        return 0;
                }
                }
            
            default:
                std::cerr << "Unknown operand type" << std::endl;
                return 0;
        }
    }
    
    // Store value in register with performance tracking
    void storeRegisterValue(size_t index, uint64_t value) {
        // Increment register write counter
        m_performanceCounters->increment(PerformanceCounterIDs::REGISTER_WRITES);
        
        // Write to register
        m_registerBank->writeRegister(index, value);
    }

    // Helper function to determine memory space based on address
    MemorySpace determineMemorySpace(uint64_t address) {
        // For now, use a simple heuristic
        // In a more sophisticated implementation, this would be based on PTX memory space specifiers
        if (address >= 0x1000 && address < 0x2000) {
            // Parameter memory range
            return MemorySpace::PARAMETER;
        } else if (address < 0x100000) {
            return MemorySpace::GLOBAL;
        } else if (address < 0x200000) {
            return MemorySpace::SHARED;
        } else {
            return MemorySpace::LOCAL;
        }
    }

    // Helper function to build CFG from decoded instructions
    void buildCFG(const std::vector<DecodedInstruction>& instructions, std::vector<std::vector<size_t>>& cfg) {
        // Build simple CFG based on instruction stream
        cfg.resize(instructions.size());
        
        for (size_t i = 0; i < instructions.size(); ++i) {
            const DecodedInstruction& instr = instructions[i];
            
            if (instr.type == InstructionTypes::BRA) {
                // For branch instructions, add target to CFG
                if (instr.sources.size() == 1 && 
                    instr.sources[0].type == OperandType::IMMEDIATE) {
                    size_t target = static_cast<size_t>(instr.sources[0].immediateValue);
                    
                    if (target < instructions.size()) {
                        cfg[i].push_back(target);
                        cfg[i].push_back(i + 1);  // Also goes to next instruction
                    }
                }
            } else {
                // For non-branch instructions, just go to next instruction
                if (i + 1 < instructions.size()) {
                    cfg[i].push_back(i + 1);
                }
            }
        }
    }
    
    // Reconstruct control flow graph from PTX
    bool buildControlFlowGraphFromPTX(const std::vector<DecodedInstruction>& instructions) {
        // For each instruction, track where branches go and where they come from
        // This information helps with divergence analysis
        
        // Clear any existing data (use the std::vector backing store)
        m_controlFlowGraphData.clear();
        
        // Initialize structures
        m_controlFlowGraphData.resize(instructions.size());
        
        // Build basic control flow graph into m_controlFlowGraphData
        for (size_t i = 0; i < instructions.size(); ++i) {
            const DecodedInstruction& instr = instructions[i];
            
            // Check if this is a branch instruction
            if (instr.type == InstructionTypes::BRA) {
                // Find branch target
                if (instr.sources.size() == 1 && 
                    instr.sources[0].type == OperandType::IMMEDIATE) {
                    // Direct branch
                    size_t targetPC = static_cast<size_t>(instr.sources[0].immediateValue);
                    
                    // Add edge from current instruction to target
                    if (targetPC < instructions.size()) {
                        m_controlFlowGraphData[i].push_back(targetPC);
                    }
                    
                    // Also add edge to next instruction (fall-through)
                    if (i + 1 < instructions.size()) {
                        m_controlFlowGraphData[i].push_back(i + 1);
                    }
                }
            } else if (instr.type == InstructionTypes::RET) {
                // Handle return instruction - no outgoing edges
            } else {
                // Normal instruction - sequential flow
                if (i + 1 < instructions.size()) {
                    m_controlFlowGraphData[i].push_back(i + 1);
                }
            }
        }
        
        // Note: If ControlFlowGraph needs to be constructed from this data,
        // add a conversion step here once the ControlFlowGraph API is known.
        return true;
    }
    
    // Getter methods for components
    PredicateHandler& getPredicateHandler() {
        return *m_predicateHandler;
    }
    
    ReconvergenceMechanism& getReconvergenceMechanism() {
        return *m_reconvergence;
    }
    
    InstructionScheduler& getInstructionScheduler() {
        return m_instructionScheduler;
    }
    
    uint32_t getCurrentCtaId() const {
        return m_currentCtaId;
    }
    
    uint32_t getCurrentGridId() const {
        return m_currentGridId;
    }
    
    void setCurrentCtaId(uint32_t id) {
        m_currentCtaId = id;
    }
    
    void setCurrentGridId(uint32_t id) {
        m_currentGridId = id;
    }
    
    DivergenceStack& getDivergenceStack() {
        return m_divergenceStack;
    }
    
    size_t getDivergenceStartCycle() const {
        return m_divergenceStartCycle;
    }
    
    void setDivergenceStartCycle(size_t cycle) {
        m_divergenceStartCycle = cycle;
    }
    
    size_t getNumDivergences() const {
        return m_numDivergences;
    }
    
    void incrementNumDivergences() {
        m_numDivergences++;
    }
    
    DivergenceStats& getDivergenceStats() {
        return m_divergenceStats;
    }
    
    ReconvergenceAlgorithm getReconvergenceAlgorithm() const {
        return m_reconvergenceAlgorithm;
    }
    
    void setReconvergenceAlgorithm(ReconvergenceAlgorithm algo) {
        m_reconvergenceAlgorithm = algo;
    }
    
    ControlFlowGraph& getControlFlowGraph() {
        return m_controlFlowGraph;
    }
    
    std::unordered_map<size_t, CFGNode*>& getPcToNode() {
        return m_pcToNode;
    }
    
    // ========================================================================
    // Multi-function execution support
    // ========================================================================
    
    // Build label address cache from program structure
    void buildLabelCache() {
        if (!m_hasProgramStructure) return;
        
        m_labelAddressCache.clear();
        
        // Add global labels
        for (const auto& [labelName, address] : m_program.symbolTable.globalLabels) {
            m_labelAddressCache[labelName] = address;
        }
        
        // Add function local labels
        for (const auto& func : m_program.functions) {
            for (const auto& [labelName, address] : func.localLabels) {
                // Prefix with function name to avoid conflicts
                std::string fullName = func.name + "::" + labelName;
                m_labelAddressCache[fullName] = address;
                // Also add without prefix for local resolution
                m_labelAddressCache[labelName] = address;
            }
        }
        
        std::cout << "Built label cache with " << m_labelAddressCache.size() << " labels" << std::endl;
    }
    
    // Resolve label to instruction address
    bool resolveLabel(const std::string& labelName, size_t& outAddress) {
        // Try current function's local labels first
        if (!m_callStack.empty()) {
            const std::string& currentFunc = m_callStack.back().functionName;
            std::string fullName = currentFunc + "::" + labelName;
            
            auto it = m_labelAddressCache.find(fullName);
            if (it != m_labelAddressCache.end()) {
                outAddress = it->second;
                return true;
            }
        }
        
        // Try global labels
        auto it = m_labelAddressCache.find(labelName);
        if (it != m_labelAddressCache.end()) {
            outAddress = it->second;
            return true;
        }
        
        return false;
    }
    
    // Validate register declarations against usage
    bool validateRegisterDeclarations() {
        if (!m_hasProgramStructure) return true;
        
        bool valid = true;
        for (const auto& func : m_program.functions) {
            // Check if register declarations are sufficient for function's instruction range
            // This is a simple validation - just check that declarations exist
            if (func.registerDeclarations.empty()) {
                std::cerr << "Warning: Function " << func.name << " has no register declarations" << std::endl;
                // Not necessarily an error - might use global registers
            }
        }
        
        return valid;
    }
    
    // Call a function by name
    bool callFunction(const std::string& funcName, const std::vector<uint64_t>& args) {
        if (!m_hasProgramStructure) {
            std::cerr << "Cannot call function: no program structure available" << std::endl;
            return false;
        }
        
        // Find the function
        auto it = m_program.symbolTable.functions.find(funcName);
        if (it == m_program.symbolTable.functions.end()) {
            std::cerr << "Function not found: " << funcName << std::endl;
            return false;
        }
        
        const PTXFunction& func = it->second;
        
        // Create call frame
        CallFrame frame;
        frame.functionName = funcName;
        frame.returnAddress = m_currentInstructionIndex + 1;
        
        // Set up parameters
        for (size_t i = 0; i < args.size() && i < func.parameters.size(); ++i) {
            const PTXParameter& param = func.parameters[i];
            frame.localParameters[param.name] = args[i];
            
            // Also write to parameter memory (using buffer-relative addressing)
            m_memorySubsystem->write<uint64_t>(MemorySpace::PARAMETER, param.offset, args[i]);
        }
        
        // Push call frame
        m_callStack.push_back(frame);
        
        // Jump to function entry
        m_currentInstructionIndex = func.startInstructionIndex;
        
        std::cout << "Calling function: " << funcName << " with " << args.size() << " arguments" << std::endl;
        
        return true;
    }
    
    // Return from current function
    bool returnFromFunction(uint64_t* returnValue = nullptr) {
        if (m_callStack.empty()) {
            // No function to return from - end execution
            m_executionComplete = true;
            return true;
        }
        
        CallFrame frame = m_callStack.back();
        m_callStack.pop_back();
        
        // Restore return address
        m_currentInstructionIndex = frame.returnAddress;
        
        std::cout << "Returning from function: " << frame.functionName << std::endl;
        
        return true;
    }
    
    // Get parameter value by name (for ld.param instruction)
    bool getParameterValue(const std::string& paramName, uint64_t& outValue) {
        if (!m_hasProgramStructure) return false;
        
        // Try current function's parameters
        if (!m_callStack.empty()) {
            const CallFrame& frame = m_callStack.back();
            auto it = frame.localParameters.find(paramName);
            if (it != frame.localParameters.end()) {
                outValue = it->second;
                return true;
            }
        }
        
        // Try symbol table
        auto it = m_program.symbolTable.parameterSymbols.find(paramName);
        if (it != m_program.symbolTable.parameterSymbols.end()) {
            const PTXParameter* param = it->second;
            // Read from parameter memory (using buffer-relative addressing)
            outValue = m_memorySubsystem->read<uint64_t>(MemorySpace::PARAMETER, param->offset);
            return true;
        }
        
        return false;
    }
    
    // Set parameter value (for st.param instruction)
    bool setParameterValue(const std::string& paramName, uint64_t value) {
        if (!m_hasProgramStructure) return false;
        
        // Update current function's parameters
        if (!m_callStack.empty()) {
            CallFrame& frame = m_callStack.back();
            frame.localParameters[paramName] = value;
        }
        
        // Update parameter memory
        auto it = m_program.symbolTable.parameterSymbols.find(paramName);
        if (it != m_program.symbolTable.parameterSymbols.end()) {
            const PTXParameter* param = it->second;
            // Write to parameter memory (using buffer-relative addressing)
            m_memorySubsystem->write<uint64_t>(MemorySpace::PARAMETER, param->offset, value);
            return true;
        }
        
        return false;
    }
    
    // Core components (using external instances)
    RegisterBank* m_registerBank;
    MemorySubsystem* m_memorySubsystem;
    
    // Program state
    std::vector<PTXInstruction> m_ptInstructions;
    std::unique_ptr<Decoder> m_decoder;
    std::vector<DecodedInstruction> m_decodedInstructions;
    size_t m_currentInstructionIndex = 0;
    bool m_executionComplete = false;
    
    // PTX Program (complete structure with functions, symbols, etc.)
    PTXProgram m_program;
    bool m_hasProgramStructure = false;
    
    // Function call stack for multi-function execution
    struct CallFrame {
        std::string functionName;
        size_t returnAddress;
        std::map<std::string, uint64_t> savedRegisters;
        std::map<std::string, uint64_t> localParameters;
    };
    std::vector<CallFrame> m_callStack;
    
    // Label resolution cache
    std::map<std::string, size_t> m_labelAddressCache;
    
    // Performance counters
    PerformanceCounters* m_performanceCounters;
    
    // Execution engine components
    std::unique_ptr<WarpScheduler> m_warpScheduler;
    std::unique_ptr<PredicateHandler> m_predicateHandler;
    std::unique_ptr<ReconvergenceMechanism> m_reconvergence;
    InstructionScheduler m_instructionScheduler;
    
    // Current execution context
    uint32_t m_currentCtaId = 0;
    uint32_t m_currentGridId = 0;
    
    // Divergence handling
    DivergenceStack m_divergenceStack;
    size_t m_divergenceStartCycle = 0;
    size_t m_numDivergences = 0;
    DivergenceStats m_divergenceStats;
    ReconvergenceAlgorithm m_reconvergenceAlgorithm = RECONVERGENCE_ALGORITHM_BASIC;
    
    // Control flow graph
    ControlFlowGraph m_controlFlowGraph;
    std::unordered_map<size_t, CFGNode*> m_pcToNode;
    std::vector<std::vector<size_t>> m_controlFlowGraphData;
};

PTXExecutor::PTXExecutor(RegisterBank& registerBank, MemorySubsystem& memorySubsystem, PerformanceCounters& performanceCounters) 
    : pImpl(std::make_unique<Impl>()) {
    // Set the external components
    pImpl->setComponents(registerBank, memorySubsystem);
    pImpl->setPerformanceCounters(performanceCounters);
}

PTXExecutor::~PTXExecutor() = default;

bool PTXExecutor::initialize(const std::vector<PTXInstruction>& ptInstructions) {
    return pImpl->initialize(ptInstructions);
}

bool PTXExecutor::initialize(const std::vector<DecodedInstruction>& decodedInstructions) {
    // Skip decoding since we already have decoded instructions
    pImpl->setDecodedInstructions(decodedInstructions);
    pImpl->setCurrentInstructionIndex(0);
    pImpl->setExecutionComplete(false);
    
    // Build control flow graph from decoded instructions
    pImpl->buildCFGFromDecodedInstructions(decodedInstructions);
    
    return true;
}

bool PTXExecutor::initialize(const PTXProgram& program) {
    // Store the complete PTX program (includes functions, parameters, symbol table)
    pImpl->m_program = program;
    pImpl->m_hasProgramStructure = true;
    
    pImpl->setDecodedInstructions(program.instructions);
    pImpl->setCurrentInstructionIndex(0);
    pImpl->setExecutionComplete(false);
    
    // Build control flow graph from decoded instructions
    pImpl->buildCFGFromDecodedInstructions(program.instructions);
    
    // Build label address cache for quick lookup
    pImpl->buildLabelCache();
    
    // Validate register declarations
    if (!pImpl->validateRegisterDeclarations()) {
        std::cerr << "Warning: Register declaration validation failed" << std::endl;
    }
    
    std::cout << "PTXExecutor initialized with PTXProgram:" << std::endl;
    std::cout << "  Version: " << program.metadata.version << std::endl;
    std::cout << "  Target: " << program.metadata.target << std::endl;
    std::cout << "  Functions: " << program.functions.size() << std::endl;
    std::cout << "  Instructions: " << program.instructions.size() << std::endl;
    std::cout << "  Entry points: " << program.entryPoints.size() << std::endl;
    
    // If there's an entry point, start execution from there
    if (!program.entryPoints.empty()) {
        size_t entryFuncIndex = program.entryPoints[0];
        if (entryFuncIndex < program.functions.size()) {
            const PTXFunction& entryFunc = program.functions[entryFuncIndex];
            pImpl->setCurrentInstructionIndex(entryFunc.startInstructionIndex);
            
            // ✅ Set all warps' PC to the entry function's start instruction
            uint32_t numWarps = pImpl->m_warpScheduler->getNumWarps();
            for (uint32_t warpId = 0; warpId < numWarps; ++warpId) {
                pImpl->m_warpScheduler->setCurrentPC(warpId, entryFunc.startInstructionIndex);
            }
            
            std::cout << "  Starting execution from entry point: " << entryFunc.name 
                     << " (PC = " << entryFunc.startInstructionIndex << ")" << std::endl;
        }
    }
    
    return true;
}

bool PTXExecutor::execute() {
    return pImpl->execute();
}

bool PTXExecutor::isExecutionComplete() const {
    return pImpl->isExecutionComplete();
}

RegisterBank& PTXExecutor::getRegisterBank() {
    return pImpl->getRegisterBank();
}

MemorySubsystem& PTXExecutor::getMemorySubsystem() {
    return pImpl->getMemorySubsystem();
}

PerformanceCounters& PTXExecutor::getPerformanceCounters() {
    return pImpl->getPerformanceCounters();
}

const std::vector<DecodedInstruction>& PTXExecutor::getDecodedInstructions() const {
    return pImpl->getDecodedInstructions();
}

bool PTXExecutor::executeSingleInstruction() {
    return pImpl->executeSingleInstruction();
}

size_t PTXExecutor::getCurrentInstructionIndex() const {
    return pImpl->getCurrentInstructionIndex();
}

WarpScheduler& PTXExecutor::getWarpScheduler() {
    return pImpl->getWarpScheduler();
}

PredicateHandler& PTXExecutor::getPredicateHandler() {
    return pImpl->getPredicateHandler();
}

ReconvergenceMechanism& PTXExecutor::getReconvergenceMechanism() {
    return pImpl->getReconvergenceMechanism();
}

InstructionScheduler& PTXExecutor::getInstructionScheduler() {
    return pImpl->getInstructionScheduler();
}

// ============================================================================
// Multi-function execution support
// ============================================================================

bool PTXExecutor::callFunction(const std::string& funcName, const std::vector<uint64_t>& args) {
    return pImpl->callFunction(funcName, args);
}

bool PTXExecutor::hasProgramStructure() const {
    return pImpl->m_hasProgramStructure;
}

size_t PTXExecutor::getCallStackDepth() const {
    return pImpl->m_callStack.size();
}

const PTXProgram& PTXExecutor::getProgram() const {
    return pImpl->m_program;
}
