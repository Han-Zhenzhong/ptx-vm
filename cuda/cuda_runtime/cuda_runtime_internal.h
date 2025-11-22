/**
 * @file cuda_runtime_internal.h
 * @brief Internal structures and utilities for libptxrt implementation
 * 
 * This header contains internal data structures that will be used
 * to implement the CUDA Runtime API functionality.
 */

#ifndef CUDA_RUNTIME_INTERNAL_H
#define CUDA_RUNTIME_INTERNAL_H

#include "cuda_runtime.h"
#include <map>
#include <string>
#include <vector>

namespace ptxrt {
namespace internal {

/**
 * @brief Structure to hold extracted PTX code from fat binary
 */
struct PTXCode {
    std::string ptx;           // PTX source code
    std::string arch;          // Target architecture (sm_XX)
    int version_major;         // PTX version major
    int version_minor;         // PTX version minor
};

/**
 * @brief Structure to hold registered kernel information
 */
struct KernelInfo {
    std::string kernel_name;   // Kernel function name in PTX
    const void* host_func;     // Host function pointer
    PTXCode* ptx_code;         // Associated PTX code
    int thread_limit;          // Thread limit (-1 if none)
};

/**
 * @brief Structure to hold fat binary information
 */
struct FatBinaryInfo {
    void* handle;                      // Fat binary handle
    std::vector<PTXCode> ptx_codes;   // All PTX variants
    std::vector<KernelInfo> kernels;  // Registered kernels
    void* raw_data;                    // Raw fat binary data
    size_t data_size;                  // Size of raw data
};

/**
 * @brief Structure to manage simulated device memory
 */
struct DeviceMemory {
    void* host_ptr;      // Actual host memory pointer
    size_t size;         // Size of allocation
    bool is_freed;       // Whether this allocation has been freed
};

/**
 * @brief Structure to hold kernel launch configuration
 */
struct LaunchConfig {
    dim3 grid_dim;
    dim3 block_dim;
    size_t shared_mem;
    cudaStream_t stream;
    std::vector<void*> args;       // Kernel arguments
    std::vector<size_t> arg_sizes; // Sizes of arguments
};

/**
 * @brief Global runtime state
 */
class RuntimeState {
public:
    // Fat binary management
    std::map<void**, FatBinaryInfo> fat_binaries;
    
    // Memory management
    std::map<void*, DeviceMemory> device_memory;
    
    // Kernel launch configuration (for <<<>>> syntax)
    LaunchConfig* current_launch_config;
    
    // Error state
    cudaError_t last_error;
    
    // Device state
    int current_device;
    int device_count;
    
    RuntimeState() 
        : current_launch_config(nullptr)
        , last_error(cudaSuccess)
        , current_device(0)
        , device_count(1) 
    {}
    
    ~RuntimeState() {
        // Cleanup
        if (current_launch_config) {
            delete current_launch_config;
        }
    }
    
    // Get singleton instance
    static RuntimeState& getInstance() {
        static RuntimeState instance;
        return instance;
    }
};

/**
 * @brief Fat binary header structures (from NVIDIA format)
 */
struct __fatBinC_Wrapper_t {
    int magic;
    int version;
    void* data;
    void* filename_or_fatbins;
};

/**
 * @brief Utility functions
 */

// Parse fat binary and extract PTX
bool parseFatBinary(void* fatCubin, FatBinaryInfo& info);

// Extract PTX from fat binary data
bool extractPTX(const void* data, size_t size, std::vector<PTXCode>& ptx_codes);

// Look up kernel by host function pointer
KernelInfo* findKernel(const void* host_func);

// Look up fat binary by handle
FatBinaryInfo* findFatBinary(void** handle);

// Allocate simulated device memory
void* allocateDeviceMemory(size_t size);

// Free simulated device memory
bool freeDeviceMemory(void* ptr);

// Copy memory (simulated host-device transfer)
bool copyMemory(void* dst, const void* src, size_t count, enum cudaMemcpyKind kind);

// Execute kernel via PTX VM
bool executeKernel(const KernelInfo& kernel, 
                   const dim3& grid_dim,
                   const dim3& block_dim,
                   void** args,
                   size_t shared_mem);

} // namespace internal
} // namespace ptxrt

#endif // CUDA_RUNTIME_INTERNAL_H
