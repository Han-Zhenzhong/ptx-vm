#include "debugger.hpp"
#include "logger.hpp"
#include <iostream>
#include <iomanip>
#include <bitset>
#include <unordered_set>
#include "execution/executor.hpp"
#include "performance_counters.hpp"
#include "execution/warp_scheduler.hpp"
#include "instruction_types.hpp"

// Private implementation class
class DebuggerImpl {
private:
    // State of the debugger
    enum class DebuggerState {
        NOT_RUNNING,      // Debugger not started
        RUNNING,          // Execution running
        STOPPED_AT_BREAKPOINT,  // Execution stopped at breakpoint
        STEPPING,         // Single stepping
    };

public:
    DebuggerImpl(PTXExecutor* executor) : m_executor(executor) {}
    
    ~DebuggerImpl() = default;
    
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
            Logger::error("Invalid memory space");
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
            std::cout << "[" << std::setw(4) << idx << "] " << getInstructionString(instructions[idx]) << std::endl;
        }
        
        // Print current instruction
        std::cout << "=> [" << std::setw(4) << currentIdx << "] " << getInstructionString(instructions[currentIdx]) << std::endl;
        
        // Print instructions after current
        for (size_t i = 1; i <= numAfter && (currentIdx + i) < instructions.size(); ++i) {
            size_t idx = currentIdx + i;
            std::cout << "[" << std::setw(4) << idx << "] " << getInstructionString(instructions[idx]) << std::endl;
        }
        
        std::cout << std::endl;
    }
    
    // Print warp execution visualization
    void printWarpVisualization() const {
        // Get warp scheduler information
        const WarpScheduler& warpScheduler = m_executor->getWarpScheduler();
        
        std::cout << "Warp Execution Visualization:" << std::endl;
        std::cout << "-----------------------------" << std::endl;
        
        // Print active warps - check each warp for active threads
        std::cout << "Active Warps: ";
        uint32_t numWarps = warpScheduler.getNumWarps();
        for (uint32_t i = 0; i < numWarps; ++i) {
            uint64_t activeThreads = warpScheduler.getActiveThreads(i);
            if (activeThreads != 0) {
                std::cout << "W" << i << " ";
            }
        }
        std::cout << std::endl;
        
        // Print thread mask for current warp
        uint32_t currentWarpId = warpScheduler.getCurrentWarp();
        uint64_t threadMask = warpScheduler.getActiveThreads(currentWarpId);
        
        std::cout << "Current Warp (W" << currentWarpId << ") Thread Mask: ";
        for (int i = 0; i < 32; ++i) {
            if (threadMask & (1 << i)) {
                std::cout << "T" << i << " ";
            }
        }
        std::cout << std::endl;
        
        // Print divergence information if available
        // Note: This functionality may need to be adapted based on actual implementation
        std::cout << "Divergence Stats:" << std::endl;
        std::cout << "  (Implementation-specific stats would be shown here)" << std::endl;
        
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
        std::cout << "  (Implementation-specific stats would be shown here)" << std::endl;
        
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
        std::cout << "  (Implementation-specific stats would be shown here)" << std::endl;
        
        std::cout << std::endl;
    }
    
private:
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
        std::cout << "Current instruction: " << getInstructionString(instr);
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
    
    // Get string representation of an instruction
    std::string getInstructionString(const DecodedInstruction& instr) const {
        // This is a simplified representation
        // In a real implementation, this would need to be more comprehensive
        std::string result = "instruction";
        // Add more detailed string representation as needed
        return result;
    }
    
    // Get name of memory space
    const char* getMemorySpaceName(MemorySpace space) {
        switch (space) {
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

Debugger::Debugger(PTXExecutor* executor) : pImpl(std::make_unique<DebuggerImpl>(executor)) {}

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