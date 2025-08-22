#include "host_api.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include "vm.hpp"
#include "cuda_binary_loader.hpp"
#include "debugger.hpp"
#include "execution/executor.hpp"  // Include for PTXExecutor complete type
#include "registers/register_bank.hpp"  // Include for RegisterBank complete type
#include "instruction_types.hpp"  // Include for InstructionTypes enum

// Private implementation class
class HostAPI::Impl {
public:
    Impl() : m_vm(nullptr), m_debugger(nullptr), m_isProgramLoaded(false) {}
    
    ~Impl() = default;

    // Initialize the VM
    bool initialize() {
        m_vm = std::make_unique<PTXVM>();
        if (!m_vm->initialize()) {
            return false;
        }
        
        m_debugger = std::make_unique<Debugger>(&m_vm->getExecutor());
        return true;
    }

    // Load a program from file
    bool loadProgram(const std::string& filename) {
        m_programFilename = filename;
        CudaBinaryLoader loader;
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
        
        return m_vm->loadAndExecuteProgram(m_programFilename);
    }

    // Step through the program
    bool step() {
        if (!m_isProgramLoaded) {
            return false;
        }
        
        return m_vm->getExecutor().executeSingleInstruction();
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
        
        const RegisterBank& registers = m_vm->getExecutor().getRegisterBank();
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
        
        const RegisterBank& registers = m_vm->getExecutor().getRegisterBank();
        std::cout << "General Purpose Registers:" << std::endl;
        for (size_t i = 0; i < registers.getNumRegisters(); ++i) {
            uint64_t value = registers.readRegister(i);
            std::cout << "  %r" << i << " = 0x" << std::hex << value << std::dec << " (" << value << ")" << std::endl;
        }
    }

    // Print predicate registers
    void printPredicateRegisters() const {
        if (!m_vm) {
            return;
        }
        
        const RegisterBank& registers = m_vm->getExecutor().getRegisterBank();
        std::cout << "Predicate Registers:" << std::endl;
        for (size_t i = 0; i < 8; ++i) {
            bool value = registers.readPredicate(i);
            std::cout << "  %p" << i << " = " << (value ? "true" : "false") << std::endl;
        }
    }

    // Print program counter
    void printProgramCounter() const {
        if (!m_vm) {
            return;
        }
        
        std::cout << "Program Counter: 0x" << std::hex << m_vm->getExecutor().getCurrentInstructionIndex() << std::dec << std::endl;
    }

    // Print memory contents
    void printMemory(uint64_t address, size_t size) const {
        if (!m_vm) {
            return;
        }
        
        std::cout << "Memory contents at 0x" << std::hex << address << std::dec << ":" << std::endl;
        for (size_t i = 0; i < size; ++i) {
            uint8_t value = 0;
            m_vm->getMemorySubsystem().read<uint8_t>(MemorySpace::GLOBAL, address + i);
            if (i % 16 == 0) {
                std::cout << std::endl << "  0x" << std::hex << (address + i) << ": ";
            }
            std::cout << std::hex << std::setfill('0') << std::setw(2) << (int)value << " ";
        }
        std::cout << std::dec << std::endl;
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
        
        m_vm->dumpExecutionStats();
    }

    // List instructions
    void listInstructions(size_t start, size_t count) const {
        // Instructions listing not yet implemented
    }

    // Print warp visualization
    void printWarpVisualization() const {
        if (!m_vm) {
            return;
        }
        
        m_vm->visualizeWarps();
    }

    // Print memory visualization
    void printMemoryVisualization() const {
        if (!m_vm) {
            return;
        }
        
        m_vm->visualizeMemory();
    }

    // Print performance counters
    void printPerformanceCounters() const {
        if (!m_vm) {
            return;
        }
        
        m_vm->visualizePerformance();
    }

private:
    std::unique_ptr<PTXVM> m_vm;
    std::unique_ptr<Debugger> m_debugger;
    std::string m_programFilename;
    bool m_isProgramLoaded;
};

HostAPI::HostAPI() : pImpl(std::make_unique<Impl>()) {}

HostAPI::~HostAPI() = default;

bool HostAPI::initialize() {
    return pImpl->initialize();
}

bool HostAPI::loadProgram(const std::string& filename) {
    return pImpl->loadProgram(filename);
}

bool HostAPI::isProgramLoaded() const {
    return pImpl->isProgramLoaded();
}

bool HostAPI::run() {
    return pImpl->run();
}

bool HostAPI::step() {
    return pImpl->step();
}

bool HostAPI::setBreakpoint(size_t address) {
    return pImpl->setBreakpoint(address);
}

bool HostAPI::setWatchpoint(uint64_t address) {
    return pImpl->setWatchpoint(address);
}

void HostAPI::printRegisters() const {
    pImpl->printRegisters();
}

void HostAPI::printAllRegisters() const {
    pImpl->printAllRegisters();
}

void HostAPI::printPredicateRegisters() const {
    pImpl->printPredicateRegisters();
}

void HostAPI::printProgramCounter() const {
    pImpl->printProgramCounter();
}

void HostAPI::printMemory(uint64_t address, size_t size) const {
    pImpl->printMemory(address, size);
}

bool HostAPI::startProfiling(const std::string& filename) {
    return pImpl->startProfiling(filename);
}

void HostAPI::dumpStatistics() const {
    pImpl->dumpStatistics();
}

void HostAPI::listInstructions(size_t start, size_t count) const {
    pImpl->listInstructions(start, count);
}

void HostAPI::printWarpVisualization() const {
    pImpl->printWarpVisualization();
}

void HostAPI::printMemoryVisualization() const {
    pImpl->printMemoryVisualization();
}

void HostAPI::printPerformanceCounters() const {
    pImpl->printPerformanceCounters();
}
