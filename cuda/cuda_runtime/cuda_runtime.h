#ifndef CUDA_RUNTIME_H
#define CUDA_RUNTIME_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// CUDA error codes
typedef enum {
    cudaSuccess = 0,
    cudaErrorInvalidValue = 1,
    cudaErrorMemoryAllocation = 2,
    cudaErrorInitializationError = 3,
    cudaErrorInvalidMemcpyDirection = 21,
    cudaErrorUnknown = 999
} cudaError_t;

// Memory copy kinds
enum cudaMemcpyKind {
    cudaMemcpyHostToHost = 0,
    cudaMemcpyHostToDevice = 1,
    cudaMemcpyDeviceToHost = 2,
    cudaMemcpyDeviceToDevice = 3,
    cudaMemcpyDefault = 4
};

// Dimension structures for grid and block
struct dim3 {
    unsigned int x, y, z;
#ifdef __cplusplus
    dim3(unsigned int x_ = 1, unsigned int y_ = 1, unsigned int z_ = 1) 
        : x(x_), y(y_), z(z_) {}
#endif
};

// Opaque handle types
typedef struct CUstream_st* cudaStream_t;
typedef struct CUevent_st* cudaEvent_t;

/**
 * @brief Memory management functions
 */

// Allocate device memory
cudaError_t cudaMalloc(void** devPtr, size_t size);

// Free device memory
cudaError_t cudaFree(void* devPtr);

// Copy memory between host and device
cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, enum cudaMemcpyKind kind);

// Set device memory to a value
cudaError_t cudaMemset(void* devPtr, int value, size_t count);

/**
 * @brief Device management functions
 */

// Synchronize device execution
cudaError_t cudaDeviceSynchronize(void);

// Reset device
cudaError_t cudaDeviceReset(void);

// Get device count
cudaError_t cudaGetDeviceCount(int* count);

// Set current device
cudaError_t cudaSetDevice(int device);

// Get current device
cudaError_t cudaGetDevice(int* device);

/**
 * @brief Error handling functions
 */

// Get error string
const char* cudaGetErrorString(cudaError_t error);

// Get last error
cudaError_t cudaGetLastError(void);

// Peek at last error without clearing it
cudaError_t cudaPeekAtLastError(void);

/**
 * @brief Kernel launch functions
 */

// Launch kernel with specified grid and block dimensions
cudaError_t cudaLaunchKernel(const void* func, dim3 gridDim, dim3 blockDim,
                             void** args, size_t sharedMem, cudaStream_t stream);

// Configure call (used by <<<>>> syntax)
cudaError_t cudaConfigureCall(dim3 gridDim, dim3 blockDim, size_t sharedMem, cudaStream_t stream);

// Setup kernel argument
cudaError_t cudaSetupArgument(const void* arg, size_t size, size_t offset);

// Launch kernel after configuration
cudaError_t cudaLaunch(const void* func);

/**
 * @brief Registration functions (called by CUDA compiler-generated code)
 */

// Fat binary handle
typedef void** fatBinaryHandle_t;

// Register fat binary containing PTX/CUBIN
void** __cudaRegisterFatBinary(void* fatCubin);

// Unregister fat binary
void __cudaUnregisterFatBinary(void** fatCubinHandle);

// Register a kernel function
void __cudaRegisterFunction(void** fatCubinHandle,
                           const char* hostFun,
                           char* deviceFun,
                           const char* deviceName,
                           int thread_limit,
                           uint3* tid,
                           uint3* bid,
                           dim3* bDim,
                           dim3* gDim,
                           int* wSize);

// Register a global variable
void __cudaRegisterVar(void** fatCubinHandle,
                      char* hostVar,
                      char* deviceAddress,
                      const char* deviceName,
                      int ext,
                      size_t size,
                      int constant,
                      int global);

// Register managed variable
void __cudaRegisterManagedVar(void** fatCubinHandle,
                              void** hostVarPtrAddress,
                              char* deviceAddress,
                              const char* deviceName,
                              int ext,
                              size_t size,
                              int constant,
                              int global);

/**
 * @brief Stream management functions
 */

// Create a stream
cudaError_t cudaStreamCreate(cudaStream_t* pStream);

// Destroy a stream
cudaError_t cudaStreamDestroy(cudaStream_t stream);

// Synchronize a stream
cudaError_t cudaStreamSynchronize(cudaStream_t stream);

/**
 * @brief Event management functions
 */

// Create an event
cudaError_t cudaEventCreate(cudaEvent_t* event);

// Destroy an event
cudaError_t cudaEventDestroy(cudaEvent_t event);

// Record an event
cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream);

// Synchronize an event
cudaError_t cudaEventSynchronize(cudaEvent_t event);

// Calculate elapsed time between events
cudaError_t cudaEventElapsedTime(float* ms, cudaEvent_t start, cudaEvent_t end);

/**
 * @brief Helper types for registration
 */

typedef struct uint3 {
    unsigned int x, y, z;
} uint3;

#ifdef __cplusplus
}
#endif

#endif // CUDA_RUNTIME_H
