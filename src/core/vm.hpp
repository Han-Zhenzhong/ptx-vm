#ifndef PTX_VM_HPP
#define PTX_VM_HPP

#include <memory>
#include "registers/register_bank.hpp"
#include "memory/memory.hpp"
#include "execution/executor.hpp"
#include "performance_counters.hpp"
#include "debugger/debugger.hpp"
#include "optimizer/register_allocator.hpp"
#include "host_api.hpp"  // Host API definitions

class PTXVM {
public:
    // Constructor/destructor
    PTXVM();
    ~PTXVM();

    // Initialize the virtual machine
    bool initialize();

    // Load and execute a PTX program
    bool loadAndExecuteProgram(const std::string& filename);

    // Get reference to the register bank
    RegisterBank& getRegisterBank() {
        return *m_registerBank;
    }

    // Get reference to the memory subsystem
    MemorySubsystem& getMemorySubsystem() {
        return *m_memorySubsystem;
    }
    
    // Get reference to the executor
    PTXExecutor& getExecutor() {
        return *m_executor;
    }
    
    // Get reference to performance counters
    PerformanceCounters& getPerformanceCounters() {
        return *m_performanceCounters;
    }
    
    // Get reference to debugger
    Debugger& getDebugger() {
        return *m_debugger;
    }
    
    // Get reference to register allocator
    RegisterAllocator& getRegisterAllocator() {
        return *m_registerAllocator;
    }
    
    // Visualization methods
    void visualizeWarps();
    void visualizeMemory();
    void visualizePerformance();
    
    // Profiling functions
    // Start profiling session
    bool startProfiling(const std::string& profileName);
    
    // Stop profiling session
    bool stopProfiling();
    
    // Dump execution statistics to console
    void dumpExecutionStats();
    
    // Dump instruction mix analysis
    void dumpInstructionMixAnalysis();
    
    // Dump memory access pattern analysis
    void dumpMemoryAccessAnalysis();
    
    // Dump warp execution analysis
    void dumpWarpExecutionAnalysis();
    
private:
    // Private implementation details
    class Impl;
    std::unique_ptr<Impl> pImpl;
    
    // Core components
    std::unique_ptr<RegisterBank> m_registerBank;  // Register management system
    std::unique_ptr<MemorySubsystem> m_memorySubsystem;  // Memory subsystem
    std::unique_ptr<PTXExecutor> m_executor;  // Execution engine
    std::unique_ptr<PerformanceCounters> m_performanceCounters;  // Performance counters
    std::unique_ptr<Debugger> m_debugger;  // Debugger
    std::unique_ptr<RegisterAllocator> m_registerAllocator;  // Register allocator
    
    // Profiling support
    std::ofstream m_profileOutputStream;
    std::string m_profileOutputFile;
    
    // Helper function to get current time as string
    std::string getCurrentTime();
};

#endif // PTX_VM_HPP