#include "host_api.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include "core/vm.hpp"
#include "host/cuda_binary_loader.hpp"
#include "debugger.hpp"

// Private implementation class
class PTXVM::Impl {
public:
    Impl() : m_vm(nullptr), m_debugger(nullptr), m_isProgramLoaded(false) {}
    
    ~Impl() = default;

    // Initialize the VM
    bool initialize() {
        m_vm = std::make_unique<VirtualMachine>();
        if (!m_vm->initialize()) {
            return false;
        }
        
        m_debugger = std::make_unique<Debugger>(m_vm->getExecutor());
        return true;
    }

    // Load a program from file
    bool loadProgram(const std::string& filename) {
        CudaBinaryLoader loader(m_vm->getExecutor());
        bool success = loader.loadBinary(filename);
        
        if (success) {
            m_isProgramLoaded = true;
        }
        
        return success;
    }

    // Check if a program is loaded
    bool isProgramLoaded() const {
        return m_isProgramLoaded;
    }

    // Run the loaded program
    bool run() {
        if (!m_isProgramLoaded) {
            return false;
        }
        
        return m_vm->execute();
    }

    // Step through the program
    bool step() {
        if (!m_isProgramLoaded) {
            return false;
        }
        
        return m_vm->getExecutor()->executeSingleInstruction();
    }

    // Set a breakpoint
    bool setBreakpoint(size_t address) {
        if (!m_debugger) {
            return false;
        }
        
        return m_debugger->setBreakpoint(address);
    }

    // Set a watchpoint
    bool setWatchpoint(uint64_t address) {
        // Watchpoints not yet implemented
        return false;
    }

    // Print registers
    void printRegisters() const {
        if (!m_vm) {
            return;
        }
        
        const RegisterBank& registers = m_vm->getExecutor()->getRegisterBank();
        std::cout << "General Purpose Registers:" << std::endl;
        for (size_t i = 0; i < 8; ++i) {
            uint64_t value = registers.readRegister(i);
            std::cout << "  %r" << i << " = 0x" << std::hex << value << std::dec << " (" << value << ")" << std::endl;
        }
    }

    // Print all registers
    void printAllRegisters() const {
        if (!m_vm) {
            return;
        }
        
        const RegisterBank& registers = m_vm->getExecutor()->getRegisterBank();
        std::cout << "All Registers:" << std::endl;
        for (size_t i = 0; i < 32; ++i) {
            uint64_t value = registers.readRegister(i);
            std::cout << "  %r" << std::setw(2) << i << " = 0x" << std::hex << std::setfill('0') << std::setw(16) << value 
                      << std::dec << std::setfill(' ') << " (" << value << ")" << std::endl;
        }
    }

    // Print predicate registers
    void printPredicateRegisters() const {
        if (!m_vm) {
            return;
        }
        
        // Predicate registers implementation would go here
        std::cout << "Predicate registers not yet implemented in this view." << std::endl;
    }

    // Print program counter
    void printProgramCounter() const {
        if (!m_vm) {
            return;
        }
        
        size_t pc = m_vm->getExecutor()->getCurrentInstructionIndex();
        std::cout << "Program Counter: " << pc << std::endl;
    }

    // Print memory contents
    void printMemory(uint64_t address, size_t size) const {
        if (!m_vm) {
            return;
        }
        
        MemorySubsystem& memory = m_vm->getExecutor()->getMemorySubsystem();
        void* buffer = memory.getMemoryBuffer(MemorySpace::GLOBAL);
        
        if (!buffer) {
            std::cout << "Unable to access memory." << std::endl;
            return;
        }
        
        uint8_t* ptr = static_cast<uint8_t*>(buffer) + address;
        
        std::cout << "Memory at 0x" << std::hex << address << std::dec << ":" << std::endl;
        for (size_t i = 0; i < size && (address + i) < memory.getMemorySize(MemorySpace::GLOBAL); ++i) {
            if (i % 8 == 0) {
                std::cout << "  0x" << std::hex << (address + i) << std::dec << ": ";
            }
            
            std::cout << std::hex << std::setfill('0') << std::setw(2) 
                      << static_cast<int>(ptr[i]) << std::dec << std::setfill(' ') << " ";
            
            if (i % 8 == 7) {
                std::cout << std::endl;
            }
        }
        
        if (size % 8 != 0) {
            std::cout << std::endl;
        }
    }

    // Start profiling
    bool startProfiling(const std::string& filename) {
        if (!m_vm) {
            return false;
        }
        
        return m_vm->startProfiling(filename);
    }

    // Dump statistics
    void dumpStatistics() const {
        if (!m_vm) {
            return;
        }
        
        const PerformanceCounters& counters = m_vm->getPerformanceCounters();
        
        std::cout << "Execution Statistics:" << std::endl;
        std::cout << "====================" << std::endl;
        std::cout << "Total Instructions: " << counters.getTotalInstructions() << std::endl;
        std::cout << "Execution Time: " << counters.getExecutionTime() << " cycles" << std::endl;
        std::cout << "Cache Hits: " << counters.getCacheHits() << std::endl;
        std::cout << "Cache Misses: " << counters.getCacheMisses() << std::endl;
        
        size_t totalCacheAccesses = counters.getCacheHits() + counters.getCacheMisses();
        if (totalCacheAccesses > 0) {
            double hitRate = (static_cast<double>(counters.getCacheHits()) / totalCacheAccesses) * 100.0;
            std::cout << "Cache Hit Rate: " << std::fixed << std::setprecision(2) << hitRate << "%" << std::endl;
        }
        
        // Print divergence statistics if available
        const auto& divergenceStats = m_vm->getExecutor()->getDivergenceStats();
        std::cout << "Divergence Statistics:" << std::endl;
        std::cout << "  Divergent Paths: " << divergenceStats.numDivergentPaths << std::endl;
        std::cout << "  Max Depth: " << divergenceStats.maxDivergenceDepth << std::endl;
        std::cout << "  Avg Rate: " << std::fixed << std::setprecision(2) 
                  << divergenceStats.averageDivergenceRate << "%" << std::endl;
    }

    // List instructions
    void listInstructions(size_t start, size_t count) const {
        if (!m_vm) {
            return;
        }
        
        const std::vector<DecodedInstruction>& instructions = m_vm->getExecutor()->getDecodedInstructions();
        
        if (start >= instructions.size()) {
            std::cout << "Start address out of range." << std::endl;
            return;
        }
        
        size_t end = std::min(start + count, instructions.size());
        
        std::cout << "Instructions " << start << " to " << (end - 1) << ":" << std::endl;
        for (size_t i = start; i < end; ++i) {
            const DecodedInstruction& instr = instructions[i];
            std::cout << "  [" << std::setw(4) << i << "] " << instr.opcode;
            
            if (!instr.dest.empty()) {
                std::cout << " " << instr.dest;
                if (!instr.sources.empty()) {
                    std::cout << ",";
                }
            }
            
            for (size_t j = 0; j < instr.sources.size(); ++j) {
                if (j > 0) std::cout << ",";
                std::cout << " " << instr.sources[j];
            }
            std::cout << std::endl;
        }
    }
    
    // Print warp execution visualization
    void printWarpVisualization() const {
        if (!m_vm) {
            return;
        }
        
        m_vm->visualizeWarps();
    }
    
    // Print memory access visualization
    void printMemoryVisualization() const {
        if (!m_vm) {
            return;
        }
        
        m_vm->visualizeMemory();
    }
    
    // Print performance counter display
    void printPerformanceCounters() const {
        if (!m_vm) {
            return;
        }
        
        m_vm->visualizePerformance();
    }

private:
    std::unique_ptr<VirtualMachine> m_vm;
    std::unique_ptr<Debugger> m_debugger;
    bool m_isProgramLoaded;
};

PTXVM::PTXVM() : pImpl(std::make_unique<Impl>()) {}

PTXVM::~PTXVM() = default;

bool PTXVM::initialize() {
    return pImpl->initialize();
}

bool PTXVM::loadProgram(const std::string& filename) {
    return pImpl->loadProgram(filename);
}

bool PTXVM::isProgramLoaded() const {
    return pImpl->isProgramLoaded();
}

bool PTXVM::run() {
    return pImpl->run();
}

bool PTXVM::step() {
    return pImpl->step();
}

bool PTXVM::setBreakpoint(size_t address) {
    return pImpl->setBreakpoint(address);
}

bool PTXVM::setWatchpoint(uint64_t address) {
    return pImpl->setWatchpoint(address);
}

void PTXVM::printRegisters() const {
    pImpl->printRegisters();
}

void PTXVM::printAllRegisters() const {
    pImpl->printAllRegisters();
}

void PTXVM::printPredicateRegisters() const {
    pImpl->printPredicateRegisters();
}

void PTXVM::printProgramCounter() const {
    pImpl->printProgramCounter();
}

void PTXVM::printMemory(uint64_t address, size_t size) const {
    pImpl->printMemory(address, size);
}

bool PTXVM::startProfiling(const std::string& filename) {
    return pImpl->startProfiling(filename);
}

void PTXVM::dumpStatistics() const {
    pImpl->dumpStatistics();
}

void PTXVM::listInstructions(size_t start, size_t count) const {
    pImpl->listInstructions(start, count);
}

void PTXVM::printWarpVisualization() const {
    pImpl->printWarpVisualization();
}

void PTXVM::printMemoryVisualization() const {
    pImpl->printMemoryVisualization();
}

void PTXVM::printPerformanceCounters() const {
    pImpl->printPerformanceCounters();
}

void PTXVM::visualizeWarps() {
    pImpl->visualizeWarps();
}

void PTXVM::visualizeMemory() {
    pImpl->visualizeMemory();
}

void PTXVM::visualizePerformance() {
    pImpl->visualizePerformance();
}
