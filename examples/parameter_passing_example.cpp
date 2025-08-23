#include "host_api.hpp"
#include <iostream>
#include <vector>

int main() {
    // Initialize the VM
    HostAPI hostAPI;
    if (!hostAPI.initialize()) {
        std::cerr << "Failed to initialize VM" << std::endl;
        return 1;
    }
    
    std::cout << "PTX VM Parameter Passing Example" << std::endl;
    std::cout << "=================================" << std::endl;
    
    // Allocate memory for input data
    const size_t dataSize = 1024;
    CUdeviceptr inputPtr;
    CUresult result = hostAPI.cuMemAlloc(&inputPtr, dataSize * sizeof(int));
    if (result != CUDA_SUCCESS) {
        std::cerr << "Failed to allocate input memory" << std::endl;
        return 1;
    }
    std::cout << "Allocated input memory at: 0x" << std::hex << inputPtr << std::dec << std::endl;
    
    // Allocate memory for output data
    CUdeviceptr outputPtr;
    result = hostAPI.cuMemAlloc(&outputPtr, dataSize * sizeof(int));
    if (result != CUDA_SUCCESS) {
        std::cerr << "Failed to allocate output memory" << std::endl;
        return 1;
    }
    std::cout << "Allocated output memory at: 0x" << std::hex << outputPtr << std::dec << std::endl;
    
    // Create some input data
    std::vector<int> inputData(dataSize);
    for (size_t i = 0; i < dataSize; ++i) {
        inputData[i] = static_cast<int>(i);
    }
    std::cout << "Created input data with " << dataSize << " elements" << std::endl;
    
    // Copy input data to VM
    result = hostAPI.cuMemcpyHtoD(inputPtr, inputData.data(), dataSize * sizeof(int));
    if (result != CUDA_SUCCESS) {
        std::cerr << "Failed to copy input data to VM" << std::endl;
        return 1;
    }
    std::cout << "Copied input data to VM memory" << std::endl;
    
    // Prepare kernel parameters
    std::cout << "Preparing kernel parameters:" << std::endl;
    std::cout << "  Input data address: 0x" << std::hex << inputPtr << std::dec << std::endl;
    std::cout << "  Output data address: 0x" << std::hex << outputPtr << std::dec << std::endl;
    std::cout << "  Data size: " << dataSize << " elements" << std::endl;
    
    // Launch a kernel with parameters
    // In a real implementation, we would load and execute an actual PTX kernel
    std::vector<void*> kernelParams;
    kernelParams.push_back(reinterpret_cast<void*>(inputPtr));
    kernelParams.push_back(reinterpret_cast<void*>(outputPtr));
    kernelParams.push_back(nullptr); // Null-terminate the array
    
    std::cout << "Launching kernel with parameters..." << std::endl;
    result = hostAPI.cuLaunchKernel(
        1, // function handle (simplified)
        1, 1, 1, // grid dimensions
        32, 1, 1, // block dimensions
        0, // shared memory
        nullptr, // stream
        kernelParams.data(), // kernel parameters
        nullptr // extra
    );
    
    if (result != CUDA_SUCCESS) {
        std::cerr << "Failed to launch kernel. Error code: " << result << std::endl;
        return 1;
    }
    
    std::cout << "Kernel launched successfully" << std::endl;
    
    // Copy results back from VM
    std::vector<int> outputData(dataSize);
    result = hostAPI.cuMemcpyDtoH(outputData.data(), outputPtr, dataSize * sizeof(int));
    if (result != CUDA_SUCCESS) {
        std::cerr << "Failed to copy output data from VM" << std::endl;
        return 1;
    }
    std::cout << "Copied output data from VM memory" << std::endl;
    
    // Display some results
    std::cout << "First 5 output elements: ";
    for (int i = 0; i < 5 && i < static_cast<int>(dataSize); ++i) {
        std::cout << outputData[i] << " ";
    }
    std::cout << std::endl;
    
    // Free memory
    hostAPI.cuMemFree(inputPtr);
    hostAPI.cuMemFree(outputPtr);
    
    std::cout << "Parameter passing example completed successfully" << std::endl;
    return 0;
}