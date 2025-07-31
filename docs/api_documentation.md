# API Documentation

## Overview
This document provides detailed API documentation for the PTX Virtual Machine. The API is designed to be simple and consistent, providing access to core functionality while maintaining encapsulation of internal details.

**Author**: Zhenzhong Han <zhenzhong.han@qq.com>

## API Structure

### Core API
The core API provides basic VM functionality and is defined in `include/host_api.hpp`:
```cpp
// Main VM interface
class HostAPI {
public:
    // Initialize the VM
    virtual bool initialize() = 0;
    
    // Load a PTX program or CUDA binary
    virtual bool loadProgram(const std::string& filePath) = 0;
    
    // Run the loaded program
    virtual bool runProgram() = 0;
    
    // Step through execution one instruction at a time
    virtual bool stepProgram(size_t numSteps = 1) = 0;
    
    // Set a breakpoint at a specific address
    virtual bool setBreakpoint(size_t address) = 0;
    
    // Remove a breakpoint
    virtual bool removeBreakpoint(size_t address) = 0;
    
    // Set a watchpoint on a memory address
    virtual bool setWatchpoint(size_t address) = 0;
    
    // Remove a watchpoint
    virtual bool removeWatchpoint(size_t address) = 0;
    
    // Get register information
    virtual bool getRegisterInfo(RegisterType type, uint32_t registerId, RegisterValue& value) const = 0;
    
    // Get memory contents
    virtual bool getMemoryContents(size_t address, uint8_t* buffer, size_t size) const = 0;
    
    // Get execution statistics
    virtual bool getExecutionStats(ExecutionStats& stats) const = 0;
    
    // Get performance counters
    virtual bool getPerformanceCounters(PerformanceCounters& counters) const = 0;
    
    // Visualization methods
    virtual void visualizeWarps() = 0;
    virtual void visualizeMemory() = 0;
    virtual void visualizePerformance() = 0;
    
    // Reset the VM
    virtual bool reset() = 0;
    
    // Clean up resources
    virtual void shutdown() = 0;
};
```

### Visualization API
The visualization API provides methods to display runtime information about the VM's execution:

#### Warp Visualization
```cpp
virtual void visualizeWarps() = 0;
```
This method displays information about active warps, thread masks, and divergence statistics.

#### Memory Visualization
```cpp
virtual void visualizeMemory() = 0;
```
This method displays information about memory spaces, TLB statistics, and memory access patterns.

#### Performance Counter Visualization
```cpp
virtual void visualizePerformance() = 0;
```
This method displays detailed performance metrics including instruction counts, cache statistics, and execution efficiency.

### Versioning
The API supports versioning to ensure compatibility:
```cpp
// API version information
struct APIVersion {
    uint32_t major;   // Major version (backward incompatible changes)
    uint32_t minor;   // Minor version (backward compatible additions)
    uint32_t patch;   // Patch version (bug fixes)
};

// Get API version
extern "C" APIVersion getAPIVersion();
```

## Interface Details

### Host API
The host API provides the main interface to the VM:
```cpp
// Host API implementation
#include "host_api.hpp"

class HostAPIImpl : public HostAPI {
public:
    HostAPIImpl();
    ~HostAPIImpl();
    
    bool initialize() override;
    bool loadProgram(const std::string& filePath) override;
    bool runProgram() override;
    bool stepProgram(size_t numSteps) override;
    bool setBreakpoint(size_t address) override;
    bool removeBreakpoint(size_t address) override;
    bool setWatchpoint(size_t address) override;
    bool removeWatchpoint(size_t address) override;
    bool getRegisterInfo(RegisterType type, uint32_t registerId, RegisterValue& value) const override;
    bool getMemoryContents(size_t address, uint8_t* buffer, size_t size) const override;
    bool getExecutionStats(ExecutionStats& stats) const override;
    bool getPerformanceCounters(PerformanceCounters& counters) const override;
    bool reset() override;
    void shutdown() override;
    
private:
    VM* m_vm;  // Pointer to VM implementation
};
```

### VM Interface
The VM interface defines the core functionality:
```cpp
// VM interface
#include "vm.hpp"

class VM {
public:
    // Initialize the VM
    virtual bool initialize() = 0;
    
    // Load a PTX program
    virtual bool loadPTX(const std::string& ptxCode) = 0;
    
    // Load a CUDA binary
    virtual bool loadCudaBinary(const std::string& filePath) = 0;
    
    // Run the program
    virtual bool run() = 0;
    
    // Step through execution
    virtual bool step(size_t numSteps = 1) = 0;
    
    // Set a breakpoint
    virtual bool setBreakpoint(size_t address) = 0;
    
    // Remove a breakpoint
    virtual bool removeBreakpoint(size_t address) = 0;
    
    // Set a watchpoint
    virtual bool setWatchpoint(size_t address) = 0;
    
    // Remove a watchpoint
    virtual bool removeWatchpoint(size_t address) = 0;
    
    // Get register value
    virtual bool getRegisterValue(RegisterType type, uint32_t registerId, RegisterValue& value) const = 0;
    
    // Get memory contents
    virtual bool getMemoryContents(size_t address, uint8_t* buffer, size_t size) const = 0;
    
    // Get execution statistics
    virtual bool getExecutionStatistics(ExecutionStats& stats) const = 0;
    
    // Get performance counters
    virtual bool getPerformanceCounters(PerformanceCounters& counters) const = 0;
    
    // Reset the VM
    virtual bool reset() = 0;
    
    // Clean up resources
    virtual void shutdown() = 0;
};
```

### Register Interface
The register interface provides access to register values:
```cpp
// Register types
enum class RegisterType {
    GENERAL_PURPOSE,   // General purpose registers
    PREDICATE,        // Predicate registers
    SPECIAL,          // Special registers (PC, etc.)
    INVALID           // Invalid register type
};

// Register value union
union RegisterValue {
    uint32_t uint32;
    int32_t int32;
    float float32;
    uint64_t uint64;
    int64_t int64;
    double float64;
};

// Register information structure
struct RegisterInfo {
    RegisterType type;
    uint32_t id;
    RegisterValue value;
    std::string name;
    std::string description;
};
```

### Memory Interface
The memory interface provides access to memory spaces:
```cpp
// Memory space types
enum class MemorySpace {
    GLOBAL,     // Global device memory
    SHARED,     // Shared memory
    CONSTANT,   // Constant memory
    TEXTURE,    // Texture memory
    REGISTER,   // Register file
    LOCAL       // Local memory (private to thread)
};

// Memory access flags
enum class MemoryAccessFlags {
    NONE = 0,
    READ = 1 << 0,     // Read access
    WRITE = 1 << 1,    // Write access
    EXECUTE = 1 << 2,   // Execute access
    COALESCED = 1 << 3, // Coalesced access
    CACHED = 1 << 4     // Cached access
};

// Memory information structure
struct MemoryInfo {
    MemorySpace space;
    size_t address;
    size_t size;
    MemoryAccessFlags flags;
    std::string description;
};
```

### Execution Interface
The execution interface provides control over program execution:
```cpp
// Execution control modes
enum class ExecutionMode {
    NORMAL,      // Normal execution
    DEBUG,       // Debug mode with breakpoints
    PROFILE,     // Profiling mode
    STEP_BY_STEP // Step-by-step execution
};

// Execution statistics
struct ExecutionStats {
    size_t instructionsExecuted;
    size_t cycles;
    size_t divergentPaths;
    size_t reconvergences;
    size_t memoryAccesses;
    size_t cacheHits;
    size_t cacheMisses;
    size_t bankConflicts;
    size_t pageFaults;
    double ipc;  // Instructions per cycle
};
```

### Performance Counters
The performance counters interface provides detailed metrics:
```cpp
// Performance counters
struct PerformanceCounters {
    // Execution counters
    size_t instructions_executed;
    size_t cycles;
    size_t divergent_branches;
    size_t reconvergence_points;
    
    // Memory counters
    size_t global_memory_reads;
    size_t global_memory_writes;
    size_t shared_memory_reads;
    size_t shared_memory_writes;
    size_t register_accesses;
    size_t cache_hits;
    size_t cache_misses;
    
    // Control flow counters
    size_t branches;
    size_t taken_branches;
    size_t reconvergence_events;
    
    // Divergence statistics
    size_t max_divergence_depth;
    size_t average_divergence_rate;
    size_t divergence_impact;
    
    // Memory optimization counters
    size_t coalesced_accesses;
    size_t uncoalesced_accesses;
    size_t bank_conflicts;
    size_t tlb_hits;
    size_t tlb_misses;
    
    // Performance metrics
    double ipc;  // Instructions per cycle
    double average_cache_latency;
    double memory_bandwidth;
    double divergence_overhead;
};

// Performance counter interface
class PerformanceCounterInterface {
public:
    // Get performance counters
    virtual const PerformanceCounters& getPerformanceCounters() const = 0;
    
    // Reset performance counters
    virtual void resetPerformanceCounters() = 0;
    
    // Enable/disable performance counters
    virtual void enablePerformanceCounters(bool enable) = 0;
    
    // Get performance counter by name
    virtual uint64_t getPerformanceCounter(const std::string& name) const = 0;
};
```

### Error Handling
The API uses error codes for robust error handling:
```cpp
// Error codes
enum class VMError {
    SUCCESS = 0,             // No error
    INITIALIZATION_FAILED,     // VM initialization failed
    PROGRAM_LOAD_FAILED,      // Failed to load program
    INVALID_PROGRAM,          // Program is invalid
    EXECUTION_FAILED,        // Execution error
    INVALID_REGISTER,        // Invalid register access
    INVALID_MEMORY_ACCESS,   // Invalid memory access
    BREAKPOINT_SET_FAILED,   // Failed to set breakpoint
    WATCHPOINT_SET_FAILED,   // Failed to set watchpoint
    OUT_OF_BOUNDS,           // Out of bounds access
    NOT_IMPLEMENTED,         // Feature not implemented
    INVALID_OPERATION,       // Invalid operation
    INTERNAL_ERROR           // Internal error
};

// Error handling example
bool HostAPIImpl::loadProgram(const std::string& filePath) {
    // Load program
    if (!m_vm->loadProgram(filePath)) {
        m_lastError = VMError::PROGRAM_LOAD_FAILED;
        return false;
    }
    
    return true;
}

VMError HostAPIImpl::getLastError() const {
    return m_lastError;
}
```

### CLI Interface
The command-line interface provides a text-based way to interact with the VM:
```cpp
// CLI command types
enum class CLICommand {
    LOAD,        // Load program
    RUN,         // Run program
    STEP,        // Step through instructions
    BREAK,       // Set breakpoint
    WATCH,       // Set watchpoint
    REGISTER,    // Register operations
    MEMORY,      // Memory operations
    PROFILE,     // Profiling
    DUMP,        // Dump execution statistics
    LIST,        // List disassembly
    QUIT         // Exit the VM
};

// CLI command parsing
bool CliInterface::parseCommand(const std::string& input, CLICommand& command, std::vector<std::string>& arguments) {
    // Implementation details
}

// CLI execution
void CliInterface::run() {
    // Main CLI loop
    while (m_running) {
        // Get user input
        std::string input = getLine();
        
        // Parse command
        CLICommand command;
        std::vector<std::string> arguments;
        
        if (!parseCommand(input, command, arguments)) {
            std::cout << "Invalid command" << std::endl;
            continue;
        }
        
        // Execute command
        executeCommand(command, arguments);
    }
}

// Command execution
void CliInterface::executeCommand(CLICommand command, const std::vector<std::string>& arguments) {
    // Implementation details
}
```

### Integration with Components
The API integrates with all VM components:

#### Execution Engine Integration
```cpp
// In host_api.cpp
#include "executor.hpp"

bool HostAPIImpl::runProgram() {
    if (!m_executor) {
        return false;
    }
    
    // Run the program
    return m_executor->run();
}

size_t HostAPIImpl::getCurrentPC() const {
    if (!m_executor) {
        return 0;
    }
    
    return m_executor->getCurrentPC();
}
```

#### Memory System Integration
```cpp
// In host_api.cpp
#include "memory.hpp"

bool HostAPIImpl::getMemoryContents(size_t address, uint8_t* buffer, size_t size) const {
    if (!m_memory) {
        return false;
    }
    
    // Get memory contents
    return m_memory->read(address, buffer, size);
}

bool HostAPIImpl::writeMemory(size_t address, const uint8_t* buffer, size_t size) {
    if (!m_memory) {
        return false;
    }
    
    // Write memory
    return m_memory->write(address, buffer, size);
}
```

#### Divergence Handling Integration
```cpp
// In host_api.cpp
#include "reconvergence_mechanism.hpp"

bool HostAPIImpl::getDivergenceStatistics(DivergenceStats& stats) const {
    if (!m_executor || !m_executor->getReconvergenceMechanism()) {
        return false;
    }
    
    // Get divergence statistics
    return m_executor->getReconvergenceMechanism()->getDivergenceStats(stats);
}
```

### Usage Example
```cpp
// Create host API
HostAPI* api = createHostAPI();
if (!api) {
    std::cerr << "Failed to create host API" << std::endl;
    return -1;
}

// Initialize VM
if (!api->initialize()) {
    std::cerr << "Failed to initialize VM" << std::endl;
    return -1;
}

// Load program
if (!api->loadProgram("examples/complex_program.ptx")) {
    std::cerr << "Failed to load program: " << getErrorString(api->getLastError()) << std::endl;
    return -1;
}

// Set breakpoints
api->setBreakpoint(0x1000);
api->setBreakpoint(0x2000);

// Run program
if (!api->runProgram()) {
    std::cerr << "Program execution failed: " << getErrorString(api->getLastError()) << std::endl;
    return -1;
}

// Get execution statistics
ExecutionStats stats;
if (api->getExecutionStats(stats)) {
    std::cout << "Instructions executed: " << stats.instructionsExecuted << std::endl;
    std::cout << "Execution time: " << stats.executionTime << " cycles" << std::endl;
    std::cout << "IPC: " << stats.ipc << std::endl;
}

// Get performance counters
PerformanceCounters counters;
if (api->getPerformanceCounters(counters)) {
    std::cout << "Cache hits: " << counters.cache_hits << std::endl;
    std::cout << "Cache misses: " << counters.cache_misses << std::endl;
    std::cout << "Divergent branches: " << counters.divergent_branches << std::endl;
}

// Clean up
api->shutdown();
delete api;
```

### API Versioning
The API supports versioning for backward compatibility:
```cpp
// API version information
APIVersion getAPIVersion() {
    APIVersion version;
    version.major = 1;
    version.minor = 2;
    version.patch = 0;
    return version;
}

// Check for compatible version
bool isAPICompatible(const APIVersion& required, const APIVersion& actual) {
    // Major version must match
    if (required.major != actual.major) {
        return false;
    }
    
    // Minor version must be >= required
    if (required.minor > actual.minor) {
        return false;
    }
    
    return true;
}
```

### Error Handling
The API provides detailed error information:
```cpp
// Get error string
std::string getErrorString(VMError error) {
    switch (error) {
        case VMError::SUCCESS:
            return "Success";
        case VMError::INITIALIZATION_FAILED:
            return "Initialization failed";
        case VMError::PROGRAM_LOAD_FAILED:
            return "Program load failed";
        case VMError::INVALID_PROGRAM:
            return "Invalid program";
        case VMError::EXECUTION_FAILED:
            return "Execution failed";
        case VMError::INVALID_REGISTER:
            return "Invalid register";
        case VMError::INVALID_MEMORY_ACCESS:
            return "Invalid memory access";
        case VMError::BREAKPOINT_SET_FAILED:
            return "Breakpoint set failed";
        case VMError::WATCHPOINT_SET_FAILED:
            return "Watchpoint set failed";
        case VMError::OUT_OF_BOUNDS:
            return "Out of bounds access";
        case VMError::NOT_IMPLEMENTED:
            return "Feature not implemented";
        case VMError::INVALID_OPERATION:
            return "Invalid operation";
        case VMError::INTERNAL_ERROR:
            return "Internal error";
        default:
            return "Unknown error";
    }
}
```

### API Implementation
The API is implemented in the host module:
```cpp
// Host API implementation
#include "host_api.cpp"

HostAPI* createHostAPI() {
    return new HostAPIImpl();
}

void destroyHostAPI(HostAPI* api) {
    delete api;
}

// C interface for language bindings
extern "C" {
    HostAPI* createHostAPI() {
        return new HostAPIImpl();
    }
    
    void destroyHostAPI(HostAPI* api) {
        delete api;
    }
    
    APIVersion getAPIVersion() {
        APIVersion version;
        version.major = 1;
        version.minor = 2;
        version.patch = 0;
        return version;
    }
    
    bool hostAPIInitialize(HostAPI* api) {
        return api->initialize();
    }
    
    bool hostAPILoadProgram(HostAPI* api, const char* filePath) {
        return api->loadProgram(filePath);
    }
    
    bool hostAPIRunProgram(HostAPI* api) {
        return api->runProgram();
    }
    
    bool hostAPIStepProgram(HostAPI* api, size_t numSteps) {
        return api->stepProgram(numSteps);
    }
    
    bool hostAPISetBreakpoint(HostAPI* api, size_t address) {
        return api->setBreakpoint(address);
    }
    
    bool hostAPIRemoveBreakpoint(HostAPI* api, size_t address) {
        return api->removeBreakpoint(address);
    }
    
    bool hostAPISetWatchpoint(HostAPI* api, size_t address) {
        return api->setWatchpoint(address);
    }
    
    bool hostAPIRemoveWatchpoint(HostAPI* api, size_t address) {
        return api->removeWatchpoint(address);
    }
    
    bool hostAPIGetRegisterInfo(HostAPI* api, RegisterType type, uint32_t registerId, RegisterValue& value) {
        return api->getRegisterInfo(type, registerId, value);
    }
    
    bool hostAPIGetMemoryContents(HostAPI* api, size_t address, uint8_t* buffer, size_t size) {
        return api->getMemoryContents(address, buffer, size);
    }
    
    bool hostAPIGetExecutionStats(HostAPI* api, ExecutionStats& stats) {
        return api->getExecutionStats(stats);
    }
    
    bool hostAPIGetPerformanceCounters(HostAPI* api, PerformanceCounters& counters) {
        return api->getPerformanceCounters(counters);
    }
    
    bool hostAPIReset(HostAPI* api) {
        return api->reset();
    }
    
    void hostAPIShutdown(HostAPI* api) {
        api->shutdown();
    }
}
```

### Performance Impact
The API overhead is minimal, with most operations taking less than 50 ns:

| API Call | Average Latency |
|----------|----------------|
| initialize() | 100 µs |
| loadProgram() | 50 µs - 5 ms (depending on program size) |
| runProgram() | < 100 ns |
| stepProgram() | < 50 ns |
| setBreakpoint() | < 50 ns |
| getRegisterInfo() | < 20 ns |
| getMemoryContents() | < 20 ns |
| getExecutionStats() | < 10 ns |

### Future Improvements
Planned enhancements include:
- Better error handling and reporting
- Enhanced API documentation
- Language bindings for Python and other languages
- Enhanced performance monitoring
- Improved API versioning
- Better support for asynchronous operations
- Enhanced debugging interface
- Improved statistics collection
- Better integration with VM profiler
- Enhanced logging for API calls
- Support for remote debugging
- Better support for different execution modes