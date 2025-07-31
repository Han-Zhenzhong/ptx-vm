#include "debugger.hpp"
#include <iostream>
#include <iomanip>
#include <bitset>
#include "execution/executor.hpp"
#include "core/performance_counters.hpp"

// Private implementation class
class Debugger::Impl {
public:
    Impl(PTXExecutor* executor) : m_executor(executor) {}
    
    ~Impl() = default;
    
    // Set a breakpoint at the specified instruction index
    bool setBreakpoint(size_t instructionIndex) {
        if (instructionIndex >= m_executor->getDecodedInstructions().size()) {
            return false;  // Invalid index
        }
        
        m_breakpoints.insert(instructionIndex);
        return true;
    }
    
    // Clear a breakpoint at the specified instruction index
    bool clearBreakpoint(size_t instructionIndex) {
        auto it = m_breakpoints.find(instructionIndex);
        if (it != m_breakpoints.end()) {
            m_breakpoints.erase(it);
            return true;
        }
        return false;
    }
    
    // Clear all breakpoints
    void clearAllBreakpoints() {
        m_breakpoints.clear();
    }
    
    // Check if there is a breakpoint at the specified index
    bool hasBreakpoint(size_t instructionIndex) const {
        return (m_breakpoints.find(instructionIndex) != m_breakpoints.end());
    }
    
    // Start execution with debugging
    bool startExecution() {
        m_currentState = DebuggerState::RUNNING;
        return executeUntilBreakpoint();
    }
    
    // Continue execution from current point
    bool continueExecution() {
        if (m_currentState == DebuggerState::STOPPED_AT_BREAKPOINT) {
            m_currentState = DebuggerState::RUNNING;
            return executeUntilBreakpoint();
        }
        return false;
    }
    
    // Step to next instruction
    bool stepInstruction() {
        if (m_currentState == DebuggerState::STOPPED_AT_BREAKPOINT || 
            m_currentState == DebuggerState::STEPPING) {
            // Execute one instruction
            bool result = m_executor->executeSingleInstruction();
            
            // Check if we hit a breakpoint
            size_t currentIdx = m_executor->getCurrentInstructionIndex();
            if (hasBreakpoint(currentIdx)) {
                m_currentState = DebuggerState::STOPPED_AT_BREAKPOINT;
            } else {
                m_currentState = DebuggerState::STEPPING;
            }
            
            // Print debug info
            printDebugInfo(currentIdx);
            
            return result;
        }
        return false;
    }
    
    // Get current instruction index
    size_t getCurrentInstructionIndex() const {
        return m_executor->getCurrentInstructionIndex();
    }
    
    // Print register state
    void printRegisters() const {
        const RegisterBank& registerBank = m_executor->getRegisterBank();
        
        std::cout << "Registers:" << std::endl;
        std::cout << "-----------" << std::endl;
        
        // Print general purpose registers
        for (size_t i = 0; i < 16; ++i) {  // Print first 16 registers
            uint64_t value = registerBank.readRegister(i);
            std::cout << "%r" << std::setw(2) << i << ": " << std::hex << "0x" << value << std::dec << " (" << value << ")" << std::endl;
        }
        
        std::cout << std::endl;
    }
    
    // Print memory state
    void printMemory(MemorySpace space, uint64_t address, size_t numWords) {
        MemorySubsystem& memorySubsystem = m_executor->getMemorySubsystem();
        void* memory = memorySubsystem.getMemoryBuffer(space);
        
        if (!memory) {
            std::cerr << "Invalid memory space" << std::endl;
            return;
        }
        
        std::cout << "Memory (space: " << getMemorySpaceName(space) << ", address: 0x" << std::hex << address << std::dec << "):" << std::endl;
        std::cout << "--------------" << std::endl;
        
        uint8_t* memoryPtr = static_cast<uint8_t*>(memory) + address;
        
        for (size_t i = 0; i < numWords; ++i) {
            uint64_t* wordPtr = reinterpret_cast<uint64_t*>(memoryPtr + i * sizeof(uint64_t));
            std::cout << "0x" << std::hex << (address + i * sizeof(uint64_t)) << ": " << std::hex << *wordPtr << std::dec << " (" << *wordPtr << ")" << std::endl;
        }
        
        std::cout << std::endl;
    }
    
    // Disassemble code around current instruction
    void disassembleCurrent(size_t numBefore, size_t numAfter) {
        const std::vector<DecodedInstruction>& instructions = m_executor->getDecodedInstructions();
        size_t currentIdx = m_executor->getCurrentInstructionIndex();
        
        std::cout << "Disassembly:" << std::endl;
        std::cout << "-------------" << std::endl;
        
        // Print instructions before current
        for (size_t i = 1; i <= numBefore && currentIdx >= i; ++i) {
            size_t idx = currentIdx - i;
            std::cout << "[" << std::setw(4) << idx << "] " << instructions[idx].opcode << std::endl;
        }
        
        // Print current instruction
        std::cout << "=> [" << std::setw(4) << currentIdx << "] " << instructions[currentIdx].opcode << std::endl;
        
        // Print instructions after current
        for (size_t i = 1; i <= numAfter && (currentIdx + i) < instructions.size(); ++i) {
            size_t idx = currentIdx + i;
            std::cout << "[" << std::setw(4) << idx << "] " << instructions[idx].opcode << std::endl;
        }
        
        std::cout << std::endl;
    }
    
    // Print warp execution visualization
    void printWarpVisualization() const {
        // Get warp scheduler information
        const WarpScheduler& warpScheduler = m_executor->getWarpScheduler();
        
        std::cout << "Warp Execution Visualization:" << std::endl;
        std::cout << "-----------------------------" << std::endl;
        
        // Print active warps
        std::cout << "Active Warps: ";
        uint64_t activeWarps = warpScheduler.getActiveWarps();
        for (int i = 0; i < 64; ++i) {
            if (activeWarps & (1ULL << i)) {
                std::cout << "W" << i << " ";
            }
        }
        std::cout << std::endl;
        
        // Print thread mask for current warp
        uint32_t currentWarpId = warpScheduler.getCurrentWarpId();
        uint64_t threadMask = warpScheduler.getActiveThreads(currentWarpId);
        
        std::cout << "Current Warp (W" << currentWarpId << ") Thread Mask: ";
        for (int i = 0; i < 32; ++i) {
            if (threadMask & (1 << i)) {
                std::cout << "T" << i << " ";
            }
        }
        std::cout << std::endl;
        
        // Print divergence information if available
        const auto& divergenceStats = m_executor->getDivergenceStats();
        std::cout << "Divergence Stats:" << std::endl;
        std::cout << "  Divergent Paths: " << divergenceStats.numDivergentPaths << std::endl;
        std::cout << "  Max Depth: " << divergenceStats.maxDivergenceDepth << std::endl;
        std::cout << "  Avg Rate: " << std::fixed << std::setprecision(2) 
                  << divergenceStats.averageDivergenceRate << "%" << std::endl;
        
        std::cout << std::endl;
    }
    
    // Print memory access visualization
    void printMemoryVisualization() const {
        // Get memory subsystem information
        const MemorySubsystem& memorySubsystem = m_executor->getMemorySubsystem();
        
        std::cout << "Memory Access Visualization:" << std::endl;
        std::cout << "----------------------------" << std::endl;
        
        // Print memory sizes
        std::cout << "Memory Spaces:" << std::endl;
        std::cout << "  Global: " << memorySubsystem.getMemorySize(MemorySpace::GLOBAL) << " bytes" << std::endl;
        std::cout << "  Shared: " << memorySubsystem.getMemorySize(MemorySpace::SHARED) << " bytes" << std::endl;
        std::cout << "  Local: " << memorySubsystem.getMemorySize(MemorySpace::LOCAL) << " bytes" << std::endl;
        std::cout << "  Parameter: " << memorySubsystem.getMemorySize(MemorySpace::PARAMETER) << " bytes" << std::endl;
        
        // Print TLB statistics if available
        std::cout << "TLB Stats:" << std::endl;
        std::cout << "  Hits: " << memorySubsystem.getTlbHits() << std::endl;
        std::cout << "  Misses: " << memorySubsystem.getTlbMisses() << std::endl;
        size_t totalTlbAccesses = memorySubsystem.getTlbHits() + memorySubsystem.getTlbMisses();
        if (totalTlbAccesses > 0) {
            double hitRate = (static_cast<double>(memorySubsystem.getTlbHits()) / totalTlbAccesses) * 100.0;
            std::cout << "  Hit Rate: " << std::fixed << std::setprecision(2) << hitRate << "%" << std::endl;
        }
        
        std::cout << std::endl;
    }
    
    // Print performance counter display
    void printPerformanceCounters() const {
        // Get performance counters
        const PerformanceCounters& perfCounters = m_executor->getPerformanceCounters();
        
        std::cout << "Performance Counters:" << std::endl;
        std::cout << "---------------------" << std::endl;
        
        // Print instruction counts
        std::cout << "Instructions:" << std::endl;
        std::cout << "  Total: " << perfCounters.getTotalInstructions() << std::endl;
        std::cout << "  Arithmetic: " << perfCounters.getArithmeticInstructions() << std::endl;
        std::cout << "  Memory: " << perfCounters.getMemoryInstructions() << std::endl;
        std::cout << "  Control Flow: " << perfCounters.getControlFlowInstructions() << std::endl;
        
        // Print execution time
        std::cout << "Execution Time: " << perfCounters.getExecutionTime() << " cycles" << std::endl;
        
        // Print cache statistics
        std::cout << "Cache Stats:" << std::endl;
        std::cout << "  Hits: " << perfCounters.getCacheHits() << std::endl;
        std::cout << "  Misses: " << perfCounters.getCacheMisses() << std::endl;
        size_t totalCacheAccesses = perfCounters.getCacheHits() + perfCounters.getCacheMisses();
        if (totalCacheAccesses > 0) {
            double hitRate = (static_cast<double>(perfCounters.getCacheHits()) / totalCacheAccesses) * 100.0;
            std::cout << "  Hit Rate: " << std::fixed << std::setprecision(2) << hitRate << "%" << std::endl;
        }
        
        std::cout << std::endl;
    }
    
private:
    // State of the debugger
    enum class DebuggerState {
        NOT_RUNNING,      // Debugger not started
        RUNNING,          // Execution running
        STOPPED_AT_BREAKPOINT,  // Execution stopped at breakpoint
        STEPPING,         // Single stepping
    };
    
    // Execute until breakpoint or completion
    bool executeUntilBreakpoint() {
        while (m_currentState == DebuggerState::RUNNING) {
            size_t currentIdx = m_executor->getCurrentInstructionIndex();
            
            // Check if this is a breakpoint
            if (hasBreakpoint(currentIdx)) {
                m_currentState = DebuggerState::STOPPED_AT_BREAKPOINT;
                std::cout << "Breakpoint hit at instruction " << currentIdx << std::endl;
                printDebugInfo(currentIdx);
                break;
            }
            
            // Execute single instruction
            if (!m_executor->executeSingleInstruction()) {
                m_currentState = DebuggerState::NOT_RUNNING;
                std::cout << "Execution completed" << std::endl;
                printDebugInfo(m_executor->getCurrentInstructionIndex());
                break;
            }
        }
        
        return (m_currentState != DebuggerState::NOT_RUNNING);
    }
    
    // Print debug information about current instruction
    void printDebugInfo(size_t currentIdx) {
        const std::vector<DecodedInstruction>& instructions = m_executor->getDecodedInstructions();
        
        if (currentIdx >= instructions.size()) {
            std::cout << "Execution completed" << std::endl;
            return;
        }
        
        // Print current instruction
        const DecodedInstruction& instr = instructions[currentIdx];
        std::cout << "Current instruction: " << instr.opcode;
        
        if (!instr.dest.empty()) {
            std::cout << " " << instr.dest;
            if (!instr.sources.empty()) {
                std::cout << ", ";
            }
        }
        
        for (size_t i = 0; i < instr.sources.size(); ++i) {
            if (i > 0) std::cout << ", ";
            std::cout << instr.sources[i];
        }
        std::cout << std::endl;
        
        // Print register state
        printRegisters();
        
        // Print memory state (show some local memory)
        printMemory(MemorySpace::LOCAL, 0, 4);
        
        std::cout << "Enter command (c=continue, s=step, r=registers, m=memory, q=quit): ";
        
        // Wait for user input if stopped at breakpoint
        if (m_currentState == DebuggerState::STOPPED_AT_BREAKPOINT) {
            char cmd;
            std::cin >> cmd;
            
            switch (cmd) {
                case 'c':  // Continue
                    m_currentState = DebuggerState::RUNNING;
                    break;
                case 's':  // Step
                    // Just execute one instruction
                    m_executor->executeSingleInstruction();
                    printDebugInfo(m_executor->getCurrentInstructionIndex());
                    break;
                case 'r':  // Show registers
                    printRegisters();
                    printDebugInfo(currentIdx);  // Show debug info again
                    break;
                case 'm':  // Show memory
                    // Show local memory again
                    printMemory(MemorySpace::LOCAL, 0, 8);
                    printDebugInfo(currentIdx);  // Show debug info again
                    break;
                case 'q':  // Quit
                    m_currentState = DebuggerState::NOT_RUNNING;
                    break;
                default:
                    std::cout << "Unknown command. Use c, s, r, m, or q" << std::endl;
                    printDebugInfo(currentIdx);  // Show debug info again
                    break;
            }
        }
    }
    
    // Get name of memory space
    const char* getMemorySpaceName(MemorySpace space) {
        switch (space) {
            case MemorySpace::GENERAL_PURPOSE: return "GENERAL";
            case MemorySpace::GLOBAL: return "GLOBAL";
            case MemorySpace::SHARED: return "SHARED";
            case MemorySpace::LOCAL: return "LOCAL";
            case MemorySpace::PARAMETER: return "PARAMETER";
            default: return "UNKNOWN";
        }
    }
    
    // Core components
    PTXExecutor* m_executor;
    
    // Debugging state
    DebuggerState m_currentState = DebuggerState::NOT_RUNNING;
    std::unordered_set<size_t> m_breakpoints;
};

Debugger::Debugger(PTXExecutor* executor) : pImpl(std::make_unique<Impl>(executor)) {}

Debugger::~Debugger() = default;

bool Debugger::setBreakpoint(size_t instructionIndex) {
    return pImpl->setBreakpoint(instructionIndex);
}

bool Debugger::clearBreakpoint(size_t instructionIndex) {
    return pImpl->clearBreakpoint(instructionIndex);
}

void Debugger::clearAllBreakpoints() {
    pImpl->clearAllBreakpoints();
}

bool Debugger::hasBreakpoint(size_t instructionIndex) const {
    return pImpl->hasBreakpoint(instructionIndex);
}

bool Debugger::startExecution() {
    return pImpl->startExecution();
}

bool Debugger::continueExecution() {
    return pImpl->continueExecution();
}

bool Debugger::stepInstruction() {
    return pImpl->stepInstruction();
}

size_t Debugger::getCurrentInstructionIndex() const {
    return pImpl->getCurrentInstructionIndex();
}

void Debugger::printRegisters() const {
    pImpl->printRegisters();
}

void Debugger::printMemory(MemorySpace space, uint64_t address, size_t numWords) {
    pImpl->printMemory(space, address, numWords);
}

void Debugger::disassembleCurrent(size_t numBefore, size_t numAfter) {
    pImpl->disassembleCurrent(numBefore, numAfter);
}

void Debugger::printWarpVisualization() const {
    pImpl->printWarpVisualization();
}

void Debugger::printMemoryVisualization() const {
    pImpl->printMemoryVisualization();
}

void Debugger::printPerformanceCounters() const {
    pImpl->printPerformanceCounters();
}