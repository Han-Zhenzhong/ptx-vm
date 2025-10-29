#include "vm.hpp"
#include <stdexcept>
#include "parser/parser.hpp"
#include <iostream>
#include "memory/memory.hpp"  // Include full definition here

// Base address in the VM address space reserved for kernel parameter memory.
// Ensure this is a CUdeviceptr so it can be assigned directly to variables of that type.
static constexpr CUdeviceptr PARAMETER_MEMORY_BASE = static_cast<CUdeviceptr>(0x10000ULL);

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
    // In a real implementation, this would:
    // 1. Allocate parameter memory space
    // 2. Copy parameter data to that space
    // 3. Set up parameter mappings for the kernel to access
    
    CUdeviceptr paramBaseAddr = PARAMETER_MEMORY_BASE;
    
    for (size_t i = 0; i < pImpl->m_kernelParameters.size(); ++i) {
        const auto& param = pImpl->m_kernelParameters[i];
        
        // Copy parameter data from device memory to parameter memory space
        try {
            for (size_t j = 0; j < param.size; ++j) {
                uint8_t value = pImpl->m_memorySubsystem->read<uint8_t>(MemorySpace::GLOBAL, param.devicePtr + j);
                pImpl->m_memorySubsystem->write<uint8_t>(MemorySpace::GLOBAL, paramBaseAddr + param.offset + j, value);
            }
        } catch (...) {
            std::cerr << "Failed to copy parameter " << i << " to parameter memory space" << std::endl;
            return false;
        }
    }
    
    // Map parameters to registers for direct access
    mapKernelParametersToRegisters();
    
    std::cout << "Set up " << pImpl->m_kernelParameters.size() << " kernel parameters in memory" << std::endl;
    return true;
}

void PTXVM::mapKernelParametersToRegisters() {
    // Map kernel parameters to registers so they can be accessed directly
    // This is a simplified implementation that maps each parameter to a register
    // In a more sophisticated implementation, this would depend on the PTX function signature
    
    RegisterBank& registerBank = pImpl->m_executor->getRegisterBank();
    
    for (size_t i = 0; i < pImpl->m_kernelParameters.size() && i < 32; ++i) {
        const auto& param = pImpl->m_kernelParameters[i];
        
        // For pointer parameters, store the pointer value in a register
        // For value parameters, we would need to read the value from parameter memory
        // In this simplified implementation, we'll store the parameter memory offset
        
        uint64_t paramValue = PARAMETER_MEMORY_BASE + param.offset;
        registerBank.writeRegister(static_cast<uint32_t>(i), paramValue);
    }
    
    std::cout << "Mapped " << pImpl->m_kernelParameters.size() << " kernel parameters to registers" << std::endl;
}

bool PTXVM::initialize() {
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
