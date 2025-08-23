#ifndef PTX_VM_HPP
#define PTX_VM_HPP

#include <memory>
#include <fstream>
#include <vector>
#include <map>
#include <string>
#include "registers/register_bank.hpp"
#include "execution/executor.hpp"
#include "performance_counters.hpp"
#include "debugger.hpp"
#include "optimizer/register_allocator.hpp"
#include "host_api.hpp"  // Host API definitions

// Explicitly define the deleter for MemorySubsystem since we're using Pimpl
// This prevents the compiler from trying to generate a default deleter which requires complete type
struct MemorySubsystemDeleter {
    void operator()(::MemorySubsystem* ptr) const;
};

// Structure to hold kernel launch parameters
struct KernelLaunchParams {
    std::string kernelName;
    unsigned int gridDimX, gridDimY, gridDimZ;
    unsigned int blockDimX, blockDimY, blockDimZ;
    unsigned int sharedMemBytes;
    std::vector<CUdeviceptr> parameters;
};

// Structure to hold kernel parameter information
struct KernelParameter {
    CUdeviceptr devicePtr;     // Device pointer to parameter data
    size_t size;               // Size of parameter data
    size_t offset;             // Offset in parameter memory space
};

class PTXVM {
public:
    // Constructor/destructor
    PTXVM();
    ~PTXVM();

    // Initialize the virtual machine
    bool initialize();

    // Load and execute a PTX program
    bool loadAndExecuteProgram(const std::string& filename);
    
    // Host API methods
    bool loadProgram(const std::string& filename);
    bool isProgramLoaded() const;
    bool run();
    bool step();
    bool setBreakpoint(size_t address);
    bool setWatchpoint(uint64_t address);
    void printRegisters() const;
    void printAllRegisters() const;
    void printPredicateRegisters() const;
    void printProgramCounter() const;
    void printMemory(uint64_t address, size_t size) const;
    void dumpStatistics() const;
    void listInstructions(size_t start, size_t count) const;
    void printWarpVisualization() const;
    void printMemoryVisualization() const;
    void printPerformanceCounters() const;

    // Kernel execution methods
    void setKernelName(const std::string& name);
    void setKernelLaunchParams(const KernelLaunchParams& params);
    bool launchKernel();
    
    // Parameter handling methods
    void setKernelParameters(const std::vector<KernelParameter>& parameters);
    bool setupKernelParameters();

    // Memory management methods
    CUdeviceptr allocateMemory(size_t size);
    bool freeMemory(CUdeviceptr ptr);
    bool copyMemoryHtoD(CUdeviceptr dst, const void* src, size_t size);
    bool copyMemoryDtoH(void* dst, CUdeviceptr src, size_t size);
    const std::map<CUdeviceptr, size_t>& getMemoryAllocations() const;

    // Get reference to the register bank
    RegisterBank& getRegisterBank() {
        return *m_registerBank;
    }

    // Get reference to the memory subsystem
    ::MemorySubsystem& getMemorySubsystem() {
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
    std::unique_ptr<::MemorySubsystem, MemorySubsystemDeleter> m_memorySubsystem;  // Memory subsystem
    std::unique_ptr<PTXExecutor> m_executor;  // Execution engine
    std::unique_ptr<PerformanceCounters> m_performanceCounters;  // Performance counters
    std::unique_ptr<Debugger> m_debugger;  // Debugger
    std::unique_ptr<RegisterAllocator> m_registerAllocator;  // Register allocator
    
    // Kernel execution state
    std::string m_currentKernelName;
    KernelLaunchParams m_kernelLaunchParams;
    std::vector<KernelParameter> m_kernelParameters;
    
    // Memory management
    std::map<CUdeviceptr, size_t> m_memoryAllocations;
    CUdeviceptr m_nextMemoryAddress;
    
    // Parameter memory space (special area for kernel parameters)
    static const CUdeviceptr PARAMETER_MEMORY_BASE = 0x1000;
    size_t m_parameterMemoryOffset;
    
    // Profiling support
    std::ofstream m_profileOutputStream;
    std::string m_profileOutputFile;
    
    // Helper function to get current time as string
    std::string getCurrentTime();
};

#endif // PTX_VM_HPP