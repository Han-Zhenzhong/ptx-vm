#include "vm.hpp"
#include <stdexcept>
#include "parser/parser.hpp"
#include <iostream>
#include "memory/memory.hpp"  // Include full definition here

// Implement the deleter for MemorySubsystem
void MemorySubsystemDeleter::operator()(::MemorySubsystem* ptr) const {
    delete ptr;
}

// Private implementation class
class PTXVM::Impl {
public:
    Impl() : m_registerBank(nullptr), m_memorySubsystem(nullptr), 
             m_executor(nullptr), m_performanceCounters(nullptr),
             m_debugger(nullptr), m_registerAllocator(nullptr) {}
    
    ~Impl() = default;

    // Core components
    std::unique_ptr<RegisterBank> m_registerBank;
    std::unique_ptr<::MemorySubsystem, MemorySubsystemDeleter> m_memorySubsystem;
    std::unique_ptr<PTXExecutor> m_executor;
    std::unique_ptr<PerformanceCounters> m_performanceCounters;
    std::unique_ptr<Debugger> m_debugger;
    std::unique_ptr<RegisterAllocator> m_registerAllocator;
    
    // Initialization state
    bool isInitialized = false;
    bool m_isProgramLoaded = false;
    std::string m_programFilename;
};

PTXVM::PTXVM() : pImpl(std::make_unique<Impl>()) {
    // Allocate registers for a simple configuration
    pImpl->m_registerBank = std::make_unique<RegisterBank>();
    if (!pImpl->m_registerBank->initialize(32)) {
        throw std::runtime_error("Failed to initialize register bank");
    }
    
    // Create memory subsystem
    pImpl->m_memorySubsystem = std::unique_ptr<::MemorySubsystem, MemorySubsystemDeleter>(new ::MemorySubsystem());
    if (!pImpl->m_memorySubsystem->initialize(1024 * 1024, 64 * 1024, 64 * 1024)) {
        throw std::runtime_error("Failed to initialize memory subsystem");
    }
    
    // Create executor
    pImpl->m_executor = std::make_unique<PTXExecutor>(*pImpl->m_registerBank, *pImpl->m_memorySubsystem);
    // Note: Executor will be initialized later with actual instructions in loadAndExecuteProgram
    
    // Create performance counters
    pImpl->m_performanceCounters = std::make_unique<PerformanceCounters>();
    
    // Create debugger
    pImpl->m_debugger = std::make_unique<Debugger>(pImpl->m_executor.get());
    
    // Create register allocator
    pImpl->m_registerAllocator = std::make_unique<RegisterAllocator>(this);
}

PTXVM::~PTXVM() = default;

bool PTXVM::initialize() {
    // Already initialized in constructor
    return true;
}

bool PTXVM::loadAndExecuteProgram(const std::string& filename) {
    // Create a parser and parse the file
    PTXParser parser;
    if (!parser.parseFile(filename)) {
        return false;
    }
    
    // Initialize executor with parsed instructions
    if (!pImpl->m_executor->initialize(parser.getInstructions())) {
        return false;
    }
    
    // Execute the program
    return pImpl->m_executor->execute();
}

bool PTXVM::run() {
    if (!pImpl->m_isProgramLoaded) {
        return false;
    }
    
    return pImpl->m_executor->execute();
}

bool PTXVM::setWatchpoint(uint64_t address) {
    // Watchpoints not yet implemented
    return false;
}

// Visualization methods
void PTXVM::visualizeWarps() {
    if (pImpl->m_debugger) {
        pImpl->m_debugger->printWarpVisualization();
    }
}

void PTXVM::visualizeMemory() {
    if (pImpl->m_debugger) {
        pImpl->m_debugger->printMemoryVisualization();
    }
}

void PTXVM::visualizePerformance() {
    if (pImpl->m_debugger) {
        pImpl->m_debugger->printPerformanceCounters();
    }
}

// Start profiling session
bool PTXVM::startProfiling(const std::string& profileName) {
    // Implementation moved to vm_profiler.cpp
    return false;  // Placeholder until we implement the actual code
}

// Stop profiling session
bool PTXVM::stopProfiling() {
    // Implementation moved to vm_profiler.cpp
    return false;  // Placeholder until we implement the actual code
}

// Get current time as string
std::string PTXVM::getCurrentTime() {
    // Implementation moved to vm_profiler.cpp
    return "";  // Placeholder until we implement the actual code
}

// Dump execution statistics to console
void PTXVM::dumpExecutionStats() {
    // Implementation moved to vm_profiler.cpp
}

// Dump instruction mix analysis
void PTXVM::dumpInstructionMixAnalysis() {
    // Implementation moved to vm_profiler.cpp
}

// Dump memory access pattern analysis
void PTXVM::dumpMemoryAccessAnalysis() {
    // Implementation moved to vm_profiler.cpp
}

// Dump warp execution analysis
void PTXVM::dumpWarpExecutionAnalysis() {
    // Implementation moved to vm_profiler.cpp
}