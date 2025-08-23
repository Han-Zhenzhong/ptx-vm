#include "host_api.hpp"
#include <iostream>
#include <vector>
#include <cstring>

int main() {
    std::cout << "PTX Virtual Machine Execution Result Demo" << std::endl;
    std::cout << "=========================================" << std::endl;
    
    // Initialize the VM
    HostAPI hostAPI;
    if (!hostAPI.initialize()) {
        std::cerr << "Failed to initialize VM" << std::endl;
        return 1;
    }
    
    // Allocate memory for input and output data
    const size_t dataSize = 1024;
    CUdeviceptr inputPtr, outputPtr;
    
    // Allocate input memory
    CUresult result = hostAPI.cuMemAlloc(&inputPtr, dataSize * sizeof(int));
    if (result != CUDA_SUCCESS) {
        std::cerr << "Failed to allocate input memory" << std::endl;
        return 1;
    }
    std::cout << "Allocated input memory at: 0x" << std::hex << inputPtr << std::dec << std::endl;
    
    // Allocate output memory
    result = hostAPI.cuMemAlloc(&outputPtr, dataSize * sizeof(int));
    if (result != CUDA_SUCCESS) {
        std::cerr << "Failed to allocate output memory" << std::endl;
        return 1;
    }
    std::cout << "Allocated output memory at: 0x" << std::hex << outputPtr << std::dec << std::endl;
    
    // Create input data
    std::vector<int> inputData(dataSize);
    for (size_t i = 0; i < dataSize; ++i) {
        inputData[i] = static_cast<int>(i);
    }
    std::cout << "Created input data with " << dataSize << " elements" << std::endl;
    std::cout << "First 5 input elements: ";
    for (int i = 0; i < 5; ++i) {
        std::cout << inputData[i] << " ";
    }
    std::cout << std::endl;
    
    // Copy input data to VM
    result = hostAPI.cuMemcpyHtoD(inputPtr, inputData.data(), dataSize * sizeof(int));
    if (result != CUDA_SUCCESS) {
        std::cerr << "Failed to copy input data to VM" << std::endl;
        return 1;
    }
    std::cout << "Copied input data to VM memory" << std::endl;
    
    // Load and execute a PTX program
    std::cout << "\nLoading and executing PTX program..." << std::endl;
    if (!hostAPI.loadProgram("examples/control_flow_example.ptx")) {
        std::cerr << "Failed to load PTX program" << std::endl;
        return 1;
    }
    
    // Run the program
    if (!hostAPI.run()) {
        std::cerr << "Failed to execute PTX program" << std::endl;
        return 1;
    }
    std::cout << "PTX program executed successfully" << std::endl;
    
    // Copy results back from VM
    std::vector<int> outputData(dataSize);
    result = hostAPI.cuMemcpyDtoH(outputData.data(), outputPtr, dataSize * sizeof(int));
    if (result != CUDA_SUCCESS) {
        std::cerr << "Failed to copy output data from VM" << std::endl;
        return 1;
    }
    std::cout << "Copied output data from VM memory" << std::endl;
    
    // Display results
    std::cout << "\nExecution Results:" << std::endl;
    std::cout << "------------------" << std::endl;
    std::cout << "First 5 output elements: ";
    for (int i = 0; i < 5; ++i) {
        std::cout << outputData[i] << " ";
    }
    std::cout << std::endl;
    
    // Show some statistics
    std::cout << "\nPerformance Statistics:" << std::endl;
    std::cout << "-----------------------" << std::endl;
    hostAPI.dumpStatistics();
    
    // Free memory
    hostAPI.cuMemFree(inputPtr);
    hostAPI.cuMemFree(outputPtr);
    
    std::cout << "\nDemo completed successfully!" << std::endl;
    return 0;
}