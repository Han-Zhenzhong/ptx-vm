#include "host_api.hpp"
#include <iostream>
#include <vector>
#include <cstring>
#include <iomanip>

int main() {
    // Initialize the Host API
    HostAPI hostAPI;
    if (!hostAPI.initialize()) {
        std::cerr << "Failed to initialize Host API" << std::endl;
        return 1;
    }
    
    std::cout << "\n=== PTX VM Parameter Passing Example ===" << std::endl;
    std::cout << "========================================\n" << std::endl;
    
    // Step 1: Load the PTX program
    std::cout << "Step 1: Loading PTX program..." << std::endl;
    if (!hostAPI.loadProgram("examples/parameter_passing_example.ptx")) {
        std::cerr << "Failed to load PTX program" << std::endl;
        return 1;
    }
    std::cout << "  ✓ Program loaded successfully\n" << std::endl;
    // Step 2: Allocate device memory
    std::cout << "Step 2: Allocating device memory..." << std::endl;
    const size_t dataSize = 256;  // Use 256 elements for testing
    
    CUdeviceptr inputPtr;
    CUresult result = hostAPI.cuMemAlloc(&inputPtr, dataSize * sizeof(int32_t));
    if (result != CUDA_SUCCESS) {
        std::cerr << "Failed to allocate input memory. Error: " << result << std::endl;
        return 1;
    }
    std::cout << "  ✓ Allocated input memory at: 0x" << std::hex << inputPtr << std::dec << std::endl;
    
    CUdeviceptr outputPtr;
    result = hostAPI.cuMemAlloc(&outputPtr, dataSize * sizeof(int32_t));
    if (result != CUDA_SUCCESS) {
        std::cerr << "Failed to allocate output memory. Error: " << result << std::endl;
        hostAPI.cuMemFree(inputPtr);
        return 1;
    }
    std::cout << "  ✓ Allocated output memory at: 0x" << std::hex << outputPtr << std::dec << std::endl;
    
    uint32_t dataSizeParam = static_cast<uint32_t>(dataSize);
    std::cout << "  Data size: " << dataSize << " elements\n" << std::endl;
    // Step 3: Prepare and copy input data
    std::cout << "Step 3: Preparing input data..." << std::endl;
    std::vector<int32_t> inputData(dataSize);
    for (size_t i = 0; i < dataSize; ++i) {
        inputData[i] = static_cast<int32_t>(i);
    }
    std::cout << "  ✓ Created input data: [0, 1, 2, ..., " << (dataSize - 1) << "]" << std::endl;
    
    result = hostAPI.cuMemcpyHtoD(inputPtr, inputData.data(), dataSize * sizeof(int32_t));
    if (result != CUDA_SUCCESS) {
        std::cerr << "Failed to copy input data to device. Error: " << result << std::endl;
        hostAPI.cuMemFree(inputPtr);
        hostAPI.cuMemFree(outputPtr);
        return 1;
    }
    std::cout << "  ✓ Copied " << dataSize << " elements to device memory\n" << std::endl;
    // Step 4: Prepare kernel parameters
    std::cout << "Step 4: Preparing kernel parameters..." << std::endl;
    std::cout << "  Parameter 0 (input_ptr):  0x" << std::hex << inputPtr << std::dec 
              << " (.u64 pointer)" << std::endl;
    std::cout << "  Parameter 1 (output_ptr): 0x" << std::hex << outputPtr << std::dec 
              << " (.u64 pointer)" << std::endl;
    std::cout << "  Parameter 2 (data_size):  " << dataSizeParam 
              << " (.u32 scalar)\n" << std::endl;
    
    // Create parameter array
    // Note: Parameters must be passed as pointers to their values
    std::vector<void*> kernelParams;
    kernelParams.push_back(&inputPtr);      // Pointer to input address
    kernelParams.push_back(&outputPtr);     // Pointer to output address
    kernelParams.push_back(&dataSizeParam); // Pointer to size value
    
    // Step 5: Launch the kernel
    std::cout << "Step 5: Launching kernel..." << std::endl;
    std::cout << "  Kernel: parameter_passing_kernel" << std::endl;
    std::cout << "  Grid dimensions:  1 x 1 x 1" << std::endl;
    std::cout << "  Block dimensions: 32 x 1 x 1" << std::endl;
    
    result = hostAPI.cuLaunchKernel(
        0,          // Function handle (0 for default entry kernel)
        1, 1, 1,    // Grid dimensions (1 block)
        32, 1, 1,   // Block dimensions (32 threads)
        0,          // Shared memory size
        nullptr,    // Stream
        kernelParams.data(),  // Kernel parameters array
        nullptr     // Extra parameters
    );
    
    if (result != CUDA_SUCCESS) {
        std::cerr << "Failed to launch kernel. Error code: " << result << std::endl;
        hostAPI.cuMemFree(inputPtr);
        hostAPI.cuMemFree(outputPtr);
        return 1;
    }
    
    std::cout << "  ✓ Kernel launched and executed successfully\n" << std::endl;
    // Step 6: Copy results back from device
    std::cout << "Step 6: Retrieving results..." << std::endl;
    std::vector<int32_t> outputData(dataSize);
    std::memset(outputData.data(), 0, dataSize * sizeof(int32_t));
    
    result = hostAPI.cuMemcpyDtoH(outputData.data(), outputPtr, dataSize * sizeof(int32_t));
    if (result != CUDA_SUCCESS) {
        std::cerr << "Failed to copy output data from device. Error: " << result << std::endl;
        hostAPI.cuMemFree(inputPtr);
        hostAPI.cuMemFree(outputPtr);
        return 1;
    }
    std::cout << "  ✓ Copied " << dataSize << " elements from device memory\n" << std::endl;
    
    // Step 7: Verify and display results
    std::cout << "Step 7: Verifying results..." << std::endl;
    std::cout << "  Expected: output[i] = input[i] * 2" << std::endl;
    std::cout << "  First 10 elements:" << std::endl;
    
    bool resultsCorrect = true;
    std::cout << "    Index | Input | Output | Expected | Status" << std::endl;
    std::cout << "    ------|-------|--------|----------|--------" << std::endl;
    
    for (int i = 0; i < 10 && i < static_cast<int>(dataSize); ++i) {
        int32_t expected = inputData[i] * 2;
        bool correct = (outputData[i] == expected);
        resultsCorrect = resultsCorrect && correct;
        
        std::cout << "    " << std::setw(5) << i << " | "
                  << std::setw(5) << inputData[i] << " | "
                  << std::setw(6) << outputData[i] << " | "
                  << std::setw(8) << expected << " | "
                  << (correct ? "✓" : "✗") << std::endl;
    }
    
    std::cout << std::endl;
    
    // Verify all results
    int correctCount = 0;
    for (size_t i = 0; i < dataSize; ++i) {
        if (outputData[i] == inputData[i] * 2) {
            correctCount++;
        }
    }
    
    std::cout << "  Results: " << correctCount << "/" << dataSize 
              << " elements correct";
    if (correctCount == static_cast<int>(dataSize)) {
        std::cout << " ✓" << std::endl;
    } else {
        std::cout << " ✗" << std::endl;
    }
    std::cout << std::endl;
    
    // Step 8: Clean up
    std::cout << "Step 8: Cleaning up..." << std::endl;
    hostAPI.cuMemFree(inputPtr);
    hostAPI.cuMemFree(outputPtr);
    std::cout << "  ✓ Device memory freed" << std::endl;
    
    std::cout << "\n=== Parameter Passing Example Complete ===" << std::endl;
    std::cout << (resultsCorrect ? "SUCCESS" : "FAILURE") << std::endl;
    
    return resultsCorrect ? 0 : 1;
}