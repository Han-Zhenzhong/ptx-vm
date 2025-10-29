#include "host_api.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <cstring>
#include "vm.hpp"
#include "cuda_binary_loader.hpp"
#include "execution/executor.hpp"  // Include for PTXExecutor complete type
#include "registers/register_bank.hpp"  // Include for RegisterBank complete type
#include "instruction_types.hpp"  // Include for InstructionTypes enum

// Private implementation class
class HostAPI::Impl {
public:
    Impl() : m_vm(nullptr), m_isProgramLoaded(false) {}
    
    ~Impl() = default;

    // Initialize the VM
    bool initialize() {
        m_vm = std::make_unique<PTXVM>();
        if (!m_vm->initialize()) {
            return false;
        }
        
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

bool HostAPI::startProfiling(const std::string& filename) {
    return pImpl->startProfiling(filename);
}

void HostAPI::dumpStatistics() const {
    pImpl->dumpStatistics();
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
