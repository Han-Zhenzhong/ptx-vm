#include "executor.hpp"
#include <stdexcept>
#include <iostream>
#include <sstream>
#include "performance_counters.hpp"
#include "warp_scheduler.hpp"  // Warp scheduler header
#include "predicate_handler.hpp"  // Predicate handler header
#include "reconvergence_mechanism.hpp"  // Reconvergence mechanism header
#include "memory/memory.hpp"  // Memory subsystem and MemorySpace

// Private implementation class
class PTXExecutor::Impl {
public:
    Impl() {
        // Initialize components
        m_registerBank = std::make_unique<RegisterBank>();
        if (!m_registerBank->initialize()) {
            throw std::runtime_error("Failed to initialize register bank");
        }
        
        m_memorySubsystem = std::make_unique<MemorySubsystem>();
        if (!m_memorySubsystem->initialize(1024 * 1024, 64 * 1024, 64 * 1024)) {
            throw std::runtime_error("Failed to initialize memory subsystem");
        }

        // Initialize warp scheduler with default configuration
        m_warpScheduler = std::make_unique<WarpScheduler>(4, 32);
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
                    bool result = executeDecodedInstruction(instr);
                    m_warpScheduler->completeInstruction(issueInfo);
                    if (!result) {
                        std::cout << "Error executing instruction" << std::endl;
                        return false;
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

    // Set register bank and memory subsystem
    void setComponents(RegisterBank& registerBank, MemorySubsystem& memorySubsystem) {
        // Reinitialize with the same parameters
        m_registerBank = std::make_unique<RegisterBank>();
        m_registerBank->initialize(registerBank.getNumRegisters());
        
        m_memorySubsystem = std::make_unique<MemorySubsystem>();
        m_memorySubsystem->initialize(
            memorySubsystem.getMemorySize(MemorySpace::GLOBAL),
            memorySubsystem.getMemorySize(MemorySpace::SHARED),
            memorySubsystem.getMemorySize(MemorySpace::LOCAL)
        );
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
    
private:
    // Execute LD for a specific memory space
    bool executeLDMemorySpace(const DecodedInstruction& instr, MemorySpace space) {
        if (instr.dest.type != OperandType::REGISTER || instr.sources.size() != 1 ||
            instr.sources[0].type != OperandType::MEMORY) {
            std::cerr << "Invalid LD instruction format" << std::endl;
            m_currentInstructionIndex++;
            return true;
        }
        uint64_t address = instr.sources[0].address;
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
        uint64_t value = m_memorySubsystem->read<uint64_t>(space, address);
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
        uint64_t address = instr.dest.address;
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
        m_memorySubsystem->write<uint64_t>(space, address, static_cast<uint64_t>(src));
        m_currentInstructionIndex++;
        return true;
    }
    // Execute a decoded instruction
    bool executeDecodedInstruction(const DecodedInstruction& instr) {
        // Check if instruction has predicate and should be skipped
        if (instr.hasPredicate) {
            // Here we get it from our predicate handler
            bool predicateValue = m_predicateHandler->evaluatePredicate(instr);
            
            if (!predicateValue) {
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
        // 这里只做简单跳转，实际应保存返回地址
        if (instr.sources.size() != 1) {
            std::cerr << "Invalid CALL instruction format" << std::endl;
            m_currentInstructionIndex++;
            return true;
        }
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
        // TODO: 保存返回地址
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
        
        // Get source operand
        int64_t src = getSourceValue(instr.sources[0]);
        
        // Store result in destination register
        storeRegisterValue(instr.dest.registerIndex, static_cast<uint64_t>(src));
        
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
        
        // Read from memory
        uint64_t address = instr.sources[0].address;
        MemorySpace space = determineMemorySpace(address);
        
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
        
        uint64_t value = m_memorySubsystem->read<uint64_t>(space, address);
        
        // Store result in destination register
        storeRegisterValue(instr.dest.registerIndex, value);
        
        // Move to next instruction
        m_currentInstructionIndex++;
        return true;
    }
    
    // Execute ST (store) instruction
    bool executeST(const DecodedInstruction& instr) {
        if (instr.sources.size() != 2 || instr.sources[0].type != OperandType::MEMORY) {
            std::cerr << "Invalid ST instruction format" << std::endl;
            m_currentInstructionIndex++;
            return true;
        }
        
        // Get source value
        int64_t src = getSourceValue(instr.sources[1]);
        
        // Write to memory
        uint64_t address = instr.sources[0].address;
        MemorySpace space = determineMemorySpace(address);
        
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
        
        m_memorySubsystem->write<uint64_t>(space, address, static_cast<uint64_t>(src));
        
        // Move to next instruction
        m_currentInstructionIndex++;
        return true;
    }
    
    // Execute BRA (branch) instruction
    bool executeBRA(const DecodedInstruction& instr) {
        if (instr.sources.size() != 1) {
            std::cerr << "Invalid BRA instruction format" << std::endl;
            m_currentInstructionIndex++;
            return true;
        }
        
        // Get branch target
        // Increment branch counter
        m_performanceCounters->increment(PerformanceCounterIDs::BRANCHES);
        
        if (instr.sources[0].type == OperandType::IMMEDIATE) {
            // Direct branch
            size_t target = static_cast<size_t>(instr.sources[0].immediateValue);
            if (target == m_currentInstructionIndex + 1) {
                // This is a sequential branch, not divergent
                m_currentInstructionIndex++;
            } else {
                // This is a non-sequential branch
                m_currentInstructionIndex = target;
                // Increment divergent branch counter
                m_performanceCounters->increment(PerformanceCounterIDs::DIVERGENT_BRANCHES);
            }
        } else if (instr.sources[0].type == OperandType::REGISTER) {
            // Indirect branch
            int64_t regValue = getSourceValue(instr.sources[0]);
            size_t target = static_cast<size_t>(regValue);
            
            if (target == m_currentInstructionIndex + 1) {
                // This is a sequential branch, not divergent
                m_currentInstructionIndex++;
            } else {
                // This is a non-sequential branch
                m_currentInstructionIndex = target;
                // Increment divergent branch counter
                m_performanceCounters->increment(PerformanceCounterIDs::DIVERGENT_BRANCHES);
            }
        } else {
            std::cerr << "Unsupported branch target type" << std::endl;
            m_currentInstructionIndex++;
            return true;
        }
        
        return true;
    }
    
    // Execute EXIT instruction
    bool executeEXIT(const DecodedInstruction& instr) {
        // Just mark execution as complete
        m_executionComplete = true;
        return true;
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
        // Get the source operand (parameter memory address)
        uint64_t paramOffset = 0;
        if (instr.sources[0].type == OperandType::IMMEDIATE) {
            paramOffset = static_cast<uint64_t>(instr.sources[0].immediateValue);
        } else if (instr.sources[0].type == OperandType::REGISTER) {
            paramOffset = impl.m_registerBank->readRegister(instr.sources[0].registerIndex);
        } else if (instr.sources[0].type == OperandType::MEMORY) {
            // [result_ptr] 这种情况，直接用偏移0（或可扩展符号表）
            paramOffset = 0;
        } else {
            std::cerr << "Invalid source operand type for LD_PARAM" << std::endl;
            impl.m_currentInstructionIndex++;
            return true;
        }
        // Calculate the actual parameter memory address
        uint64_t paramAddress = 0x1000 + paramOffset;  // PARAMETER_MEMORY_BASE = 0x1000
        // Read from parameter memory
        uint64_t value = impl.m_memorySubsystem->read<uint64_t>(MemorySpace::GLOBAL, paramAddress);
        // Store result in destination register
        impl.storeRegisterValue(instr.dest.registerIndex, value);
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
        
        // Calculate the actual parameter memory address
        uint64_t paramAddress = 0x1000 + paramOffset;  // PARAMETER_MEMORY_BASE = 0x1000
        
        // Write to parameter memory
        impl.m_memorySubsystem->write<uint64_t>(MemorySpace::GLOBAL, paramAddress, srcValue);
        
        // Move to next instruction
        impl.m_currentInstructionIndex++;
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
    
    // Handle synchronization points
    void handleSynchronization(const DecodedInstruction& instruction) {
        // At a synchronization point, we need to check for divergence
        // and determine which threads can proceed
        
        // Get the predicate handler
        PredicateHandler* predicateHandler = m_predicateHandler.get();
        
        // If there's an active divergence stack entry, we may need to reconverge
        size_t joinPC;
        uint64_t savedMask;
        uint64_t savedDivergentMask;
        
        if (!predicateHandler->isDivergenceStackEmpty()) {
            // Pop the top divergence point
            if (predicateHandler->popDivergencePoint(joinPC, savedMask, savedDivergentMask)) {
                // Check if we've reached the join point
                // Note: We would need access to currentPC to do this properly
                // For now, we'll just push it back
                predicateHandler->pushDivergencePoint(joinPC, savedMask, savedDivergentMask);
            }
        }
    }
    
    // Reconstruct control flow graph from PTX
    bool buildControlFlowGraphFromPTX(const std::vector<DecodedInstruction>& instructions) {
        // For each instruction, track where branches go and where they come from
        // This information helps with divergence analysis
        
        // Clear any existing data
        m_controlFlowGraph.clear();
        
        // Initialize structures
        m_controlFlowGraph.resize(instructions.size());
        
        // Build basic control flow graph
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
                        m_controlFlowGraph[i].push_back(targetPC);
                    }
                    
                    // Also add edge to next instruction (fall-through)
                    if (i + 1 < instructions.size()) {
                        m_controlFlowGraph[i].push_back(i + 1);
                    }
                }
            } else if (instr.type == InstructionTypes::RET) {
                // Handle return instruction - no outgoing edges
            } else {
                // Normal instruction - sequential flow
                if (i + 1 < instructions.size()) {
                    m_controlFlowGraph[i].push_back(i + 1);
                }
            }
        }
        
        return true;
    }
    
    // Core components
    std::unique_ptr<RegisterBank> m_registerBank;
    std::unique_ptr<MemorySubsystem> m_memorySubsystem;
    
    // Program state
    std::vector<PTXInstruction> m_ptInstructions;
    std::unique_ptr<Decoder> m_decoder;
    std::vector<DecodedInstruction> m_decodedInstructions;
    size_t m_currentInstructionIndex = 0;
    bool m_executionComplete = false;
    
    // Performance counters
    PerformanceCounters* m_performanceCounters;
    
    // Execution engine components
    std::unique_ptr<WarpScheduler> m_warpScheduler;
    std::unique_ptr<PredicateHandler> m_predicateHandler;
    std::unique_ptr<ReconvergenceMechanism> m_reconvergence;  // Reconvergence mechanism
    
    // Control flow graph
    std::vector<std::vector<size_t>> m_controlFlowGraph;
};

PTXExecutor::PTXExecutor() : pImpl(std::make_unique<Impl>()), 
                             m_performanceCounters(pImpl->getPerformanceCounters()) {}

PTXExecutor::PTXExecutor(RegisterBank& registerBank, MemorySubsystem& memorySubsystem) 
    : pImpl(std::make_unique<Impl>()), 
      m_performanceCounters(pImpl->getPerformanceCounters()) {
    // Override the default register bank and memory subsystem with the provided ones
    pImpl->setComponents(registerBank, memorySubsystem);
}

PTXExecutor::PTXExecutor(RegisterBank& registerBank, MemorySubsystem& memorySubsystem, PerformanceCounters& performanceCounters) 
    : pImpl(std::make_unique<Impl>()), 
      m_performanceCounters(performanceCounters) {
    // Override the default register bank and memory subsystem with the provided ones
    // pImpl->setComponents(registerBank, memorySubsystem);
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
