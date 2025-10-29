#include "host_api.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <cstring>
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
    
    // Memory management functions
    CUresult cuMemAlloc(CUdeviceptr* dptr, size_t bytesize) {
        if (!m_vm || !dptr) {
            return CUDA_ERROR_INVALID_VALUE;
        }
        
        try {
            // Allocate memory in the VM's global memory space
            MemorySubsystem& memorySubsystem = m_vm->getMemorySubsystem();
            
            // Find a suitable location in global memory
            // For simplicity, we'll just allocate at the end of existing allocations
            // In a more sophisticated implementation, we would use a proper memory allocator
            
            // Get the global memory space
            void* globalMem = memorySubsystem.getMemoryBuffer(MemorySpace::GLOBAL);
            size_t globalMemSize = memorySubsystem.getMemorySize(MemorySpace::GLOBAL);
            
            // For now, we'll use a simple approach and allocate at a fixed offset
            // A real implementation would need a proper memory management system
            static uint64_t allocationOffset = 0x10000; // Start allocating at 64KB
            
            if (allocationOffset + bytesize > globalMemSize) {
                return CUDA_ERROR_OUT_OF_MEMORY;
            }
            
            *dptr = allocationOffset;
            allocationOffset += bytesize;
            
            // Ensure proper alignment (8-byte alignment for most types)
            allocationOffset = (allocationOffset + 7) & ~7;
            
            return CUDA_SUCCESS;
        } catch (...) {
            return CUDA_ERROR_UNKNOWN;
        }
    }
    
    CUresult cuMemFree(CUdeviceptr dptr) {
        // In a simple implementation, we don't actually free memory
        // A more sophisticated implementation would track allocations and free them
        return CUDA_SUCCESS;
    }
    
    CUresult cuMemcpyHtoD(CUdeviceptr dstDevice, const void* srcHost, size_t ByteCount) {
        if (!m_vm || !srcHost) {
            return CUDA_ERROR_INVALID_VALUE;
        }
        
        try {
            MemorySubsystem& memorySubsystem = m_vm->getMemorySubsystem();
            
            // Copy data from host to device (VM) memory
            for (size_t i = 0; i < ByteCount; ++i) {
                const uint8_t* src = static_cast<const uint8_t*>(srcHost);
                memorySubsystem.write<uint8_t>(MemorySpace::GLOBAL, dstDevice + i, src[i]);
            }
            
            return CUDA_SUCCESS;
        } catch (...) {
            return CUDA_ERROR_UNKNOWN;
        }
    }
    
    CUresult cuMemcpyDtoH(void* dstHost, CUdeviceptr srcDevice, size_t ByteCount) {
        if (!m_vm || !dstHost) {
            return CUDA_ERROR_INVALID_VALUE;
        }
        
        try {
            MemorySubsystem& memorySubsystem = m_vm->getMemorySubsystem();
            
            // Copy data from device (VM) memory to host
            for (size_t i = 0; i < ByteCount; ++i) {
                uint8_t value = memorySubsystem.read<uint8_t>(MemorySpace::GLOBAL, srcDevice + i);
                uint8_t* dst = static_cast<uint8_t*>(dstHost);
                dst[i] = value;
            }
            
            return CUDA_SUCCESS;
        } catch (...) {
            return CUDA_ERROR_UNKNOWN;
        }
    }
    
    // Kernel launch function
    CUresult cuLaunchKernel(
        CUfunction f,
        unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
        unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
        unsigned int sharedMemBytes, CUstream hStream, void** kernelParams, void** extra
    ) {
        if (!m_vm || !m_isProgramLoaded) {
            return CUDA_ERROR_INVALID_VALUE;
        }

        try {
            std::cout << "Launching kernel with function handle: " << f << std::endl;
            std::cout << "Grid dimensions: " << gridDimX << " x " << gridDimY << " x " << gridDimZ << std::endl;
            std::cout << "Block dimensions: " << blockDimX << " x " << blockDimY << " x " << blockDimZ << std::endl;

            PTXExecutor& executor = m_vm->getExecutor();
            
            // 🔧 修复：将 kernelParams 复制到参数内存
            if (kernelParams != nullptr && executor.hasProgramStructure()) {
                const PTXProgram& program = executor.getProgram();
                
                if (!program.functions.empty()) {
                    const PTXFunction& entryFunc = program.functions[0];
                    MemorySubsystem& mem = executor.getMemorySubsystem();
                    
                    std::cout << "Setting up " << entryFunc.parameters.size() << " kernel parameters..." << std::endl;
                    
                    // 将每个参数复制到参数内存
                    size_t offset = 0;
                    for (size_t i = 0; i < entryFunc.parameters.size(); ++i) {
                        const PTXParameter& param = entryFunc.parameters[i];
                        
                        // kernelParams[i] 指向实际的参数数据
                        if (kernelParams[i] != nullptr) {
                            std::cout << "  Parameter " << i << " (" << param.name << "): "
                                      << "type=" << param.type << ", size=" << param.size 
                                      << ", offset=" << offset << std::endl;
                            
                            // 将参数数据复制到参数内存 (基址 0x1000)
                            const uint8_t* paramData = static_cast<const uint8_t*>(kernelParams[i]);
                            for (size_t j = 0; j < param.size; ++j) {
                                mem.write<uint8_t>(MemorySpace::PARAMETER, 
                                                  0x1000 + offset + j, 
                                                  paramData[j]);
                            }
                        }
                        
                        offset += param.size;
                    }
                    
                    std::cout << "Kernel parameters successfully copied to parameter memory" << std::endl;
                }
            }

            // 设置grid/block维度
            // TODO: 传递给 warp scheduler

            // 执行内核
            bool success = m_vm->run();
            return success ? CUDA_SUCCESS : CUDA_ERROR_LAUNCH_FAILED;
            
        } catch (const std::exception& e) {
            std::cerr << "Kernel launch error: " << e.what() << std::endl;
            return CUDA_ERROR_UNKNOWN;
        } catch (...) {
            return CUDA_ERROR_UNKNOWN;
        }
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

// Implement CUDA-like API functions
CUresult HostAPI::cuInit(unsigned int flags) {
    // Initialization is handled in the constructor
    return CUDA_SUCCESS;
}

CUresult HostAPI::cuDeviceGet(CUdevice* device, int ordinal) {
    if (!device || ordinal != 0) {
        return CUDA_ERROR_INVALID_DEVICE;
    }
    
    *device = 0;
    return CUDA_SUCCESS;
}

CUresult HostAPI::cuDeviceGetCount(int* count) {
    if (!count) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    
    *count = 1; // We only have one virtual device
    return CUDA_SUCCESS;
}

CUresult HostAPI::cuDeviceGetName(char* name, int len, CUdevice device) {
    if (!name || len <= 0 || device != 0) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    
    const char* deviceName = "PTX Virtual Machine";
    int nameLen = strlen(deviceName);
    
    if (len <= nameLen) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    
    strcpy(name, deviceName);
    return CUDA_SUCCESS;
}

CUresult HostAPI::cuDeviceComputeCapability(int* major, int* minor, CUdevice device) {
    if (!major || !minor || device != 0) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    
    *major = 6; // PTX 6.1 support
    *minor = 1;
    return CUDA_SUCCESS;
}

CUresult HostAPI::cuCtxCreate(CUcontext* pctx, unsigned int flags, CUdevice device) {
    if (!pctx || device != 0) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    
    *pctx = 1; // Simple context handle
    return CUDA_SUCCESS;
}

CUresult HostAPI::cuCtxDestroy(CUcontext ctx) {
    if (!ctx) {
        return CUDA_ERROR_INVALID_CONTEXT;
    }
    
    return CUDA_SUCCESS;
}

CUresult HostAPI::cuCtxPushCurrent(CUcontext ctx) {
    // Context stack not implemented in this simple VM
    return CUDA_SUCCESS;
}

CUresult HostAPI::cuCtxPopCurrent(CUcontext* pctx) {
    if (pctx) {
        *pctx = 1;
    }
    return CUDA_SUCCESS;
}

CUresult HostAPI::cuModuleLoad(CUmodule* module, const char* fname) {
    if (!module || !fname) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    
    // In our simple implementation, we'll just store an identifier
    // A real implementation would parse and load the module
    *module = 1;  // Using a simple numeric identifier
    return CUDA_SUCCESS;
}

CUresult HostAPI::cuModuleLoadData(CUmodule* module, const void* image) {
    if (!module || !image) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    
    // Not implemented in this simple VM
    *module = 1;  // Using a simple numeric identifier
    return CUDA_SUCCESS;
}

CUresult HostAPI::cuModuleUnload(CUmodule module) {
    if (!module) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    
    // In a real implementation, this would clean up module resources
    return CUDA_SUCCESS;
}

CUresult HostAPI::cuModuleGetFunction(CUfunction* hfunc, CUmodule module, const char* name) {
    if (!hfunc || !module || !name) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    
    // In our simple implementation, we'll just store a function identifier
    // In a real implementation, this would map to actual function data
    *hfunc = 1;  // Using a simple numeric identifier
    return CUDA_SUCCESS;
}

CUresult HostAPI::cuMemAlloc(CUdeviceptr* dptr, size_t bytesize) {
    return pImpl->cuMemAlloc(dptr, bytesize);
}

CUresult HostAPI::cuMemFree(CUdeviceptr dptr) {
    return pImpl->cuMemFree(dptr);
}

CUresult HostAPI::cuMemcpyHtoD(CUdeviceptr dstDevice, const void* srcHost, size_t ByteCount) {
    return pImpl->cuMemcpyHtoD(dstDevice, srcHost, ByteCount);
}

CUresult HostAPI::cuMemcpyDtoH(void* dstHost, CUdeviceptr srcDevice, size_t ByteCount) {
    return pImpl->cuMemcpyDtoH(dstHost, srcDevice, ByteCount);
}

CUresult HostAPI::cuLaunchKernel(CUfunction f,
                               unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
                               unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
                               unsigned int sharedMemBytes, CUstream hStream, void** kernelParams, void** extra) {
    return pImpl->cuLaunchKernel(f, gridDimX, gridDimY, gridDimZ, 
                                blockDimX, blockDimY, blockDimZ,
                                sharedMemBytes, hStream, kernelParams, extra);
}

CUresult HostAPI::cuStreamCreate(CUstream* phStream, unsigned int flags) {
    if (!phStream) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    
    *phStream = reinterpret_cast<CUstream>(0x1);
    return CUDA_SUCCESS;
}

CUresult HostAPI::cuStreamQuery(CUstream hStream) {
    return CUDA_SUCCESS;
}

CUresult HostAPI::cuStreamSynchronize(CUstream hStream) {
    return CUDA_SUCCESS;
}

CUresult HostAPI::cuStreamDestroy(CUstream hStream) {
    return CUDA_SUCCESS;
}

CUresult HostAPI::cuProfilerStart() {
    // Profiling functionality not implemented
    return CUDA_SUCCESS;
}

CUresult HostAPI::cuProfilerStop() {
    // Profiling functionality not implemented
    return CUDA_SUCCESS;
}

void HostAPI::visualizeWarps() {
    pImpl->printWarpVisualization();
}

void HostAPI::visualizeMemory() {
    pImpl->printMemoryVisualization();
}

void HostAPI::visualizePerformance() {
    pImpl->printPerformanceCounters();
}
