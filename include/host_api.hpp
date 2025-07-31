#ifndef HOST_API_HPP
#define HOST_API_HPP

#include <cstdint>
#include <string>
#include <vector>
#include "vm.hpp"

typedef uint32_t CUdevice;
typedef uint32_t CUcontext;
typedef uint32_t CUmodule;
typedef uint32_t CUfunction;
typedef uint64_t CUdeviceptr;

// CUDA result codes
typedef enum {
    CUDA_SUCCESS = 0,
    CUDA_ERROR_INVALID_VALUE = 1,
    CUDA_ERROR_OUT_OF_MEMORY = 2,
    CUDA_ERROR_NOT_INITIALIZED = 3,
    CUDA_ERROR_DEINITIALIZED = 4,
    CUDA_ERROR_NO_DEVICE = 100,
    CUDA_ERROR_INVALID_DEVICE = 101,
    CUDA_ERROR_INVALID_IMAGE = 200,
    CUDA_ERROR_INVALID_CONTEXT = 201,
    CUDA_ERROR_CONTEXT_ALREADY_CURRENT = 202,
    CUDA_ERROR_MAP_FAILED = 205,
    CUDA_ERROR_UNMAP_FAILED = 206,
    CUDA_ERROR_ARRAY_IS_MAPPED = 207,
    CUDA_ERROR_ALREADY_MAPPED = 208,
    CUDA_ERROR_NO_BINARY_FOR_GPU = 209,
    CUDA_ERROR_ALREADY_ACQUIRED = 210,
    CUDA_ERROR_NOT_MAPPED = 211,
    CUDA_ERROR_NOT_MAPPED_AS_ARRAY = 212,
    CUDA_ERROR_NOT_MAPPED_AS_POINTER = 213,
    CUDA_ERROR_ECC_UNCORRECTABLE = 214,
    CUDA_ERROR_UNSUPPORTED_LIMIT = 215,
    CUDA_ERROR_CONTEXT_ALREADY_IN_USE = 216,
    CUDA_ERROR_INVALID_SOURCE = 300,
    CUDA_ERROR_FILE_NOT_FOUND = 301,
    CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND = 302,
    CUDA_ERROR_SHARED_OBJECT_INIT_FAILED = 303,
    CUDA_ERROR_OPERATING_SYSTEM = 304,
    CUDA_ERROR_INVALID_HANDLE = 400,
    CUDA_ERROR_NOT_FOUND = 500,
    CUDA_ERROR_NOT_READY = 600,
    CUDA_ERROR_LAUNCH_FAILED = 700,
    CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES = 701,
    CUDA_ERROR_LAUNCH_TIMEOUT = 702,
    CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING = 703,
    CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED = 704,
    CUDA_ERROR_PEER_ACCESS_NOT_ENABLED = 705,
    CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE = 708,
    CUDA_ERROR_CONTEXT_IS_DESTROYED = 709,
    CUDA_ERROR_ASSERT = 710,
    CUDA_ERROR_TOO_MANY_PEERS = 711,
    CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED = 712,
    CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED = 713,
    CUDA_ERROR_HARDWARE_STACK_ERROR = 714,
    CUDA_ERROR_ILLEGAL_INSTRUCTION = 715,
    CUDA_ERROR_MISALIGNED_ADDRESS = 716,
    CUDA_ERROR_INVALID_ADDRESS_SPACE = 717,
    CUDA_ERROR_INVALID_PC = 718,
    CUDA_ERROR_LAUNCH_FAILURE = 719,
    CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE = 720,
    CUDA_ERROR_NOT_PERMITTED = 800,
    CUDA_ERROR_NOT_SUPPORTED = 801,
    CUDA_ERROR_UNKNOWN = 999
} CUresult;

class HostAPI {
public:
    // Constructor/destructor
    HostAPI();
    ~HostAPI();

    // Initialize the host API
    bool initialize();

    // CUDA-like API functions
    
    // Device management
    CUresult cuInit(unsigned int flags);
    CUresult cuDeviceGet(CUdevice* device, int ordinal);
    CUresult cuDeviceGetCount(int* count);
    CUresult cuDeviceGetName(char* name, int len, CUdevice device);
    CUresult cuDeviceComputeCapability(int* major, int* minor, CUdevice device);
    
    // Context management
    CUresult cuCtxCreate(CUcontext* pctx, unsigned int flags, CUdevice device);
    CUresult cuCtxDestroy(CUcontext ctx);
    CUresult cuCtxPushCurrent(CUcontext ctx);
    CUresult cuCtxPopCurrent(CUcontext* pctx);
    
    // Module management (PTX programs)
    CUresult cuModuleLoad(CUmodule* module, const char* fname);
    CUresult cuModuleLoadData(CUmodule* module, const void* image);
    CUresult cuModuleUnload(CUmodule module);
    CUresult cuModuleGetFunction(CUfunction* hfunc, CUmodule module, const char* name);
    
    // Memory management
    CUresult cuMemAlloc(CUdeviceptr* dptr, size_t bytesize);
    CUresult cuMemFree(CUdeviceptr dptr);
    CUresult cuMemcpyHtoD(CUdeviceptr dstDevice, const void* srcHost, size_t ByteCount);
    CUresult cuMemcpyDtoH(void* dstHost, CUdeviceptr srcDevice, size_t ByteCount);
    
    // Kernel execution
    CUresult cuLaunchKernel(CUfunction f,
                          unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
                          unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
                          unsigned int sharedMemBytes, CUstream hStream, void** kernelParams, void** extra);
    
    // Stream management
    typedef void* CUstream;
    CUresult cuStreamCreate(CUstream* phStream, unsigned int flags);
    CUresult cuStreamQuery(CUstream hStream);
    CUresult cuStreamSynchronize(CUstream hStream);
    CUresult cuStreamDestroy(CUstream hStream);
    
    // Profiling and debugging
    CUresult cuProfilerStart();
    CUresult cuProfilerStop();
    
    // Visualization methods
    void visualizeWarps();
    void visualizeMemory();
    void visualizePerformance();
    
private:
    // Private implementation details
    class Impl;
    std::unique_ptr<Impl> pImpl;
};

// Simplified API for testing
namespace SimpleHostAPI {
    // Initialize the virtual machine
    bool initializeVM();
    
    // Load a PTX program
    bool loadProgram(const std::string& filename);
    
    // Allocate memory on the VM
    CUdeviceptr allocateMemory(size_t size);
    
    // Free memory on the VM
    void freeMemory(CUdeviceptr ptr);
    
    // Copy memory to VM
    bool copyToVM(CUdeviceptr dest, const void* src, size_t size);
    
    // Copy memory from VM
    bool copyFromVM(void* dest, CUdeviceptr src, size_t size);
    
    // Launch a kernel
    bool launchKernel(const std::string& kernelName, 
                     const std::vector<CUdeviceptr>& arguments,
                     unsigned int gridDimX = 1, unsigned int gridDimY = 1, unsigned int gridDimZ = 1,
                     unsigned int blockDimX = 1, unsigned int blockDimY = 1, unsigned int blockDimZ = 1);
    
    // Get performance counters
    const PerformanceCounters& getPerformanceCounters();
    
    // Print warp execution visualization
    void printWarpVisualization();
    
    // Print memory access visualization
    void printMemoryVisualization();
    
    // Print performance counter display
    void printPerformanceCounters();
}

#endif // HOST_API_HPP