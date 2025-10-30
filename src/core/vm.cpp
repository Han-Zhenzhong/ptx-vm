#include "vm.hpp"
#include <stdexcept>
#include "parser/parser.hpp"
#include <iostream>
#include "memory/memory.hpp"  // Include full definition here

// Explicitly define the deleter for MemorySubsystem since we're using Pimpl
// This prevents the compiler from trying to generate a default deleter which requires complete type
struct MemorySubsystemDeleter {
    void operator()(::MemorySubsystem* ptr) const;
};

// Implement the deleter for MemorySubsystem
void MemorySubsystemDeleter::operator()(::MemorySubsystem* ptr) const {
    delete ptr;
}

// Private implementation class
class PTXVM::Impl {
public:
    Impl() : m_registerBank(std::make_unique<RegisterBank>()),
             m_memorySubsystem(new ::MemorySubsystem(), MemorySubsystemDeleter()),
             m_performanceCounters(std::make_unique<PerformanceCounters>()),
             m_currentKernelName(""),
             m_nextMemoryAddress(0x10000), // Start allocating at 64KB
             m_parameterMemoryOffset(0),
             isInitialized(false),
             m_isProgramLoaded(false) {}
    
    ~Impl() = default;

    // Core components
    std::unique_ptr<RegisterBank> m_registerBank;
    std::unique_ptr<::MemorySubsystem, MemorySubsystemDeleter> m_memorySubsystem;
    std::unique_ptr<PTXExecutor> m_executor;
    std::unique_ptr<PerformanceCounters> m_performanceCounters;
    std::unique_ptr<Debugger> m_debugger;
    std::unique_ptr<RegisterAllocator> m_registerAllocator;
    
    // Kernel execution state
    std::string m_currentKernelName;
    KernelLaunchParams m_kernelLaunchParams;
    std::vector<KernelParameter> m_kernelParameters;
    
    // Memory management
    std::map<CUdeviceptr, size_t> m_memoryAllocations;
    CUdeviceptr m_nextMemoryAddress;
    
    // Parameter memory space
    size_t m_parameterMemoryOffset;
    
    // Initialization state
    bool isInitialized;
    bool m_isProgramLoaded;
    std::string m_programFilename;
    
    // Getter methods
    RegisterBank& getRegisterBank() {
        return *m_registerBank;
    }
    
    ::MemorySubsystem& getMemorySubsystem() {
        return *m_memorySubsystem;
    }
    
    PTXExecutor& getExecutor() {
        return *m_executor;
    }
    
    PerformanceCounters& getPerformanceCounters() {
        return *m_performanceCounters;
    }
    
    Debugger& getDebugger() {
        return *m_debugger;
    }
    
    RegisterAllocator& getRegisterAllocator() {
        return *m_registerAllocator;
    }
};

PTXVM::PTXVM() : pImpl(std::make_unique<Impl>()) {}

PTXVM::~PTXVM() = default;

void PTXVM::setKernelName(const std::string& name) {
    pImpl->m_currentKernelName = name;
}

void PTXVM::setKernelLaunchParams(const KernelLaunchParams& params) {
    pImpl->m_kernelLaunchParams = params;
}

void PTXVM::setKernelParameters(const std::vector<KernelParameter>& parameters) {
    pImpl->m_kernelParameters = parameters;
}

bool PTXVM::setupKernelParameters() {
    // Setup kernel parameters in VM memory
    // Parameters are written to the PARAMETER memory space at base address 0x1000
    // For this implementation:
    // - param.devicePtr contains the actual parameter VALUE (e.g., a pointer or scalar)
    // - param.size is the size of the parameter in bytes
    // - param.offset is the offset within parameter memory
        
    for (size_t i = 0; i < pImpl->m_kernelParameters.size(); ++i) {
        const auto& param = pImpl->m_kernelParameters[i];
        
        // Write parameter value directly to parameter memory space
        // param.devicePtr contains the value to write (e.g., a pointer address)
        try {
            // Write the parameter value byte-by-byte to parameter memory
            // Note: The memory subsystem uses buffer-relative addressing (starting at 0),
            // so we use param.offset directly, not paramBaseAddr + param.offset
            CUdeviceptr paramValue = param.devicePtr;
            for (size_t j = 0; j < param.size; ++j) {
                uint8_t byte = static_cast<uint8_t>((paramValue >> (j * 8)) & 0xFF);
                pImpl->m_memorySubsystem->write<uint8_t>(MemorySpace::PARAMETER, 
                                                         param.offset + j, 
                                                         byte);
            }
        } catch (const std::exception& e) {
            std::cerr << "Failed to copy parameter " << i << " to parameter memory space: " 
                      << e.what() << std::endl;
            return false;
        } catch (...) {
            std::cerr << "Failed to copy parameter " << i << " to parameter memory space" << std::endl;
            return false;
        }
    }
    
    // Note: We do NOT pre-map parameters to registers.
    // PTX code will use ld.param instructions to load parameters dynamically.
    // The old mapKernelParametersToRegisters() was conceptually incorrect.
    
    std::cout << "Set up " << pImpl->m_kernelParameters.size() << " kernel parameters in memory" << std::endl;
    return true;
}

void PTXVM::mapKernelParametersToRegisters() {
    // DEPRECATED: This function is no longer needed.
    // 
    // In real PTX execution, parameters are NOT pre-mapped to registers.
    // Instead, PTX code uses ld.param instructions to load parameters on-demand.
    // 
    // Example from simple_math_example.ptx:
    //   ld.param.u64 %r0, [result_ptr];
    // 
    // This instruction will:
    // 1. Look up "result_ptr" in the parameter space
    // 2. Read the value from PARAMETER memory at the appropriate offset
    // 3. Store it in register %r0
    // 
    // The previous implementation was incorrect because:
    // - It wrote PARAMETER_MEMORY_BASE + offset to registers, but the ld.param
    //   instruction expects the actual parameter VALUE, not an address
    // - It arbitrarily mapped parameters to registers r0, r1, r2... which
    //   conflicts with how PTX code actually declares and uses registers
    
    // This function is kept for API compatibility but does nothing
}

bool PTXVM::initialize() {
    // Initialize register bank first (allocate register arrays)
    // Allocate enough registers for typical PTX programs
    // PTX typically uses: %r0-%rN (integer), %f0-%fN (float), %d0-%dN (double), %p0-%p7 (predicate)
    // We allocate 256 registers for each type to support most programs
    if (!pImpl->m_registerBank->initialize(256, 256)) {
        std::cerr << "Failed to initialize register bank" << std::endl;
        return false;
    }
    
    // Initialize memory subsystem (allocate memory spaces)
    // Default sizes: 1MB global, 64KB shared, 64KB local, plus parameter memory
    if (!pImpl->m_memorySubsystem->initialize(
            1024 * 1024,  // 1 MB global memory
            64 * 1024,    // 64 KB shared memory
            64 * 1024)) { // 64 KB local memory
        std::cerr << "Failed to initialize memory subsystem" << std::endl;
        return false;
    }
    
    // Initialize the executor with register bank and memory subsystem
    pImpl->m_executor = std::make_unique<PTXExecutor>(*pImpl->m_registerBank, *pImpl->m_memorySubsystem, *pImpl->m_performanceCounters);
    
    // Initialize the debugger with the executor
    pImpl->m_debugger = std::make_unique<Debugger>(pImpl->m_executor.get());
    
    // Initialize the register allocator with this VM
    pImpl->m_registerAllocator = std::make_unique<RegisterAllocator>(this);
    pImpl->m_registerAllocator->initialize();
    
    // Initialize the register allocator with default parameters
    // 16 physical registers, 1 warp, 32 threads per warp
    if (!pImpl->m_registerAllocator->allocateRegisters(16, 1, 32)) {
        return false;
    }
    
    return true;
}

bool PTXVM::loadProgram(const std::string& filename) {
    pImpl->m_programFilename = filename;

    // Create a parser and parse the file
    PTXParser parser;
    if (!parser.parseFile(filename)) {
        std::cerr << "Failed to parse PTX file: " << filename << std::endl;
        std::cerr << "Error: " << parser.getErrorMessage() << std::endl;
        return false;
    }
    
    // Get the complete PTX program
    const PTXProgram& program = parser.getProgram();
    
    // Initialize executor with the complete PTX program (not just instructions)
    if (!pImpl->m_executor->initialize(program)) {
        std::cerr << "Failed to initialize executor with PTX program" << std::endl;
        return false;
    }

    pImpl->m_isProgramLoaded = true;
    std::cout << "Successfully loaded PTX program from: " << filename << std::endl;
    
    // Execute the program
    return true;
}

bool PTXVM::run() {
    if (!pImpl->m_isProgramLoaded) {
        std::cerr << "No program loaded" << std::endl;
        return false;
    }
    
    // ✅ Setup kernel parameters before execution
    // This writes parameters to parameter memory (0x1000 base)
    // and maps them to registers
    if (!pImpl->m_kernelParameters.empty()) {
        if (!setupKernelParameters()) {
            std::cerr << "Failed to setup kernel parameters" << std::endl;
            return false;
        }
    }
    
    // Execute the program
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

CUdeviceptr PTXVM::allocateMemory(size_t size) {
    // Align to 8-byte boundary
    size = (size + 7) & ~7;
    
    CUdeviceptr address = pImpl->m_nextMemoryAddress;
    pImpl->m_memoryAllocations[address] = size;
    pImpl->m_nextMemoryAddress += size;
    
    return address;
}

bool PTXVM::freeMemory(CUdeviceptr ptr) {
    auto it = pImpl->m_memoryAllocations.find(ptr);
    if (it != pImpl->m_memoryAllocations.end()) {
        pImpl->m_memoryAllocations.erase(it);
        return true;
    }
    return false;
}

bool PTXVM::copyMemoryHtoD(CUdeviceptr dst, const void* src, size_t size) {
    if (!src) return false;
    
    try {
        for (size_t i = 0; i < size; ++i) {
            const uint8_t* srcBytes = static_cast<const uint8_t*>(src);
            pImpl->m_memorySubsystem->write<uint8_t>(MemorySpace::GLOBAL, dst + i, srcBytes[i]);
        }
        return true;
    } catch (...) {
        return false;
    }
}

bool PTXVM::copyMemoryDtoH(void* dst, CUdeviceptr src, size_t size) {
    if (!dst) return false;
    
    try {
        for (size_t i = 0; i < size; ++i) {
            uint8_t value = pImpl->m_memorySubsystem->read<uint8_t>(MemorySpace::GLOBAL, src + i);
            uint8_t* dstBytes = static_cast<uint8_t*>(dst);
            dstBytes[i] = value;
        }
        return true;
    } catch (...) {
        return false;
    }
}

const std::map<CUdeviceptr, size_t>& PTXVM::getMemoryAllocations() const {
    return pImpl->m_memoryAllocations;
}

bool PTXVM::isProgramLoaded() const {
    return pImpl->m_isProgramLoaded;
}

bool PTXVM::hasProgram() const {
    return pImpl->m_isProgramLoaded && pImpl->m_executor != nullptr;
}

RegisterBank& PTXVM::getRegisterBank() {
    return pImpl->getRegisterBank();
}

::MemorySubsystem& PTXVM::getMemorySubsystem() {
    return pImpl->getMemorySubsystem();
}

PTXExecutor& PTXVM::getExecutor() {
    return pImpl->getExecutor();
}

PerformanceCounters& PTXVM::getPerformanceCounters() {
    return pImpl->getPerformanceCounters();
}

Debugger& PTXVM::getDebugger() {
    return pImpl->getDebugger();
}

RegisterAllocator& PTXVM::getRegisterAllocator() {
    return pImpl->getRegisterAllocator();
}
