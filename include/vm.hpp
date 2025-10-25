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

// Forward declarations to avoid circular includes
typedef uint64_t CUdeviceptr;

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
    void mapKernelParametersToRegisters();

    // Memory management methods
    CUdeviceptr allocateMemory(size_t size);
    bool freeMemory(CUdeviceptr ptr);
    bool copyMemoryHtoD(CUdeviceptr dst, const void* src, size_t size);
    bool copyMemoryDtoH(void* dst, CUdeviceptr src, size_t size);
    const std::map<CUdeviceptr, size_t>& getMemoryAllocations() const;

    // Get reference to the register bank
    RegisterBank& getRegisterBank();

    // Get reference to the memory subsystem
    ::MemorySubsystem& getMemorySubsystem();
    
    // Get reference to the executor
    PTXExecutor& getExecutor();
    
    // Get reference to performance counters
    PerformanceCounters& getPerformanceCounters();
    
    // Get reference to debugger
    Debugger& getDebugger();
    
    // Get reference to register allocator
    RegisterAllocator& getRegisterAllocator();
    
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
    
    // Parameter memory space (special area for kernel parameters)
    static const CUdeviceptr PARAMETER_MEMORY_BASE = 0x1000;
    
    // Helper function to get current time as string
    std::string getCurrentTime();
};

#endif // PTX_VM_HPP