#include "host_api.hpp"
#include <iostream>
#include <vector>
#include <cstring>
#include <iomanip>

int main() {
    std::cout << "\n=== PTX Virtual Machine Execution Result Demo ===" << std::endl;
    std::cout << "================================================\n" << std::endl;
    
    // Step 1: Initialize the VM
    std::cout << "Step 1: Initializing VM..." << std::endl;
    HostAPI hostAPI;
    if (!hostAPI.initialize()) {
        std::cerr << "Failed to initialize VM" << std::endl;
        return 1;
    }
    std::cout << "  ✓ VM initialized successfully\n" << std::endl;
    
    // Step 2: Load PTX program
    std::cout << "Step 2: Loading PTX program..." << std::endl;
    if (!hostAPI.loadProgram("examples/control_flow_example.ptx")) {
        std::cerr << "Failed to load PTX program" << std::endl;
        return 1;
    }
    std::cout << "  ✓ PTX program loaded: control_flow_example.ptx\n" << std::endl;
    
    // Step 3: Allocate device memory
    std::cout << "Step 3: Allocating device memory..." << std::endl;
    const size_t dataSize = 256;  // Use 256 elements for the demo
    CUdeviceptr inputPtr, resultPtr;
    
    // Allocate input memory
    CUresult result = hostAPI.cuMemAlloc(&inputPtr, dataSize * sizeof(int32_t));
    if (result != CUDA_SUCCESS) {
        std::cerr << "Failed to allocate input memory. Error: " << result << std::endl;
        return 1;
    }
    std::cout << "  ✓ Allocated input memory at: 0x" << std::hex << inputPtr << std::dec << std::endl;
    
    // Allocate result memory
    result = hostAPI.cuMemAlloc(&resultPtr, dataSize * sizeof(int32_t));
    if (result != CUDA_SUCCESS) {
        std::cerr << "Failed to allocate result memory. Error: " << result << std::endl;
        hostAPI.cuMemFree(inputPtr);
        return 1;
    }
    std::cout << "  ✓ Allocated result memory at: 0x" << std::hex << resultPtr << std::dec << "\n" << std::endl;
    
    // Step 4: Prepare and copy input data
    std::cout << "Step 4: Preparing input data..." << std::endl;
    std::vector<int32_t> inputData(dataSize);
    for (size_t i = 0; i < dataSize; ++i) {
        inputData[i] = static_cast<int32_t>(i + 1);  // Values from 1 to 256
    }
    std::cout << "  ✓ Created input data: [1, 2, 3, ..., " << dataSize << "]" << std::endl;
    std::cout << "    First 5 elements: ";
    for (int i = 0; i < 5; ++i) {
        std::cout << inputData[i] << " ";
    }
    std::cout << std::endl;
    
    result = hostAPI.cuMemcpyHtoD(inputPtr, inputData.data(), dataSize * sizeof(int32_t));
    if (result != CUDA_SUCCESS) {
        std::cerr << "Failed to copy input data to device. Error: " << result << std::endl;
        hostAPI.cuMemFree(inputPtr);
        hostAPI.cuMemFree(resultPtr);
        return 1;
    }
    std::cout << "  ✓ Copied " << dataSize << " elements to device memory\n" << std::endl;
    
    // Step 5: Launch kernel
    std::cout << "Step 5: Launching kernel..." << std::endl;
    std::cout << "  Kernel: control_flow_kernel" << std::endl;
    std::cout << "  Grid dimensions:  1 x 1 x 1" << std::endl;
    std::cout << "  Block dimensions: 32 x 1 x 1" << std::endl;
    std::cout << "  Parameters:" << std::endl;
    std::cout << "    - input_ptr:  0x" << std::hex << inputPtr << std::dec << " (.u64 pointer)" << std::endl;
    std::cout << "    - result_ptr: 0x" << std::hex << resultPtr << std::dec << " (.u64 pointer)" << std::endl;
    
    // Prepare kernel parameters
    std::vector<void*> kernelParams;
    kernelParams.push_back(&inputPtr);   // input_ptr parameter
    kernelParams.push_back(&resultPtr);  // result_ptr parameter
    
    // Launch the kernel using cuLaunchKernel
    result = hostAPI.cuLaunchKernel(
        0,          // Function handle (0 for default entry kernel)
        1, 1, 1,    // Grid dimensions (1 block)
        32, 1, 1,   // Block dimensions (32 threads)
        0,          // Shared memory size
        nullptr,    // Stream
        kernelParams.data(),  // Kernel parameters
        nullptr     // Extra parameters
    );
    
    if (result != CUDA_SUCCESS) {
        std::cerr << "Failed to launch kernel. Error code: " << result << std::endl;
        hostAPI.cuMemFree(inputPtr);
        hostAPI.cuMemFree(resultPtr);
        return 1;
    }
    std::cout << "  ✓ Kernel executed successfully\n" << std::endl;
    
    // Step 6: Retrieve results
    std::cout << "Step 6: Retrieving results from device..." << std::endl;
    std::vector<int32_t> resultData(dataSize);
    std::memset(resultData.data(), 0, dataSize * sizeof(int32_t));
    
    result = hostAPI.cuMemcpyDtoH(resultData.data(), resultPtr, dataSize * sizeof(int32_t));
    if (result != CUDA_SUCCESS) {
        std::cerr << "Failed to copy result data from device. Error: " << result << std::endl;
        hostAPI.cuMemFree(inputPtr);
        hostAPI.cuMemFree(resultPtr);
        return 1;
    }
    std::cout << "  ✓ Copied " << dataSize << " elements from device memory\n" << std::endl;
    
    // Step 7: Display and verify results
    std::cout << "Step 7: Verifying execution results..." << std::endl;
    std::cout << "  Control flow kernel performs: result = sum of (input * 5)" << std::endl;
    std::cout << "  First 10 results:" << std::endl;
    std::cout << "    Index | Input | Result | Expected | Status" << std::endl;
    std::cout << "    ------|-------|--------|----------|--------" << std::endl;
    
    int correctCount = 0;
    for (int i = 0; i < 10 && i < static_cast<int>(dataSize); ++i) {
        // The control flow kernel adds the input value 5 times (loop from 0 to 5)
        int32_t expected = inputData[i] * 5;
        bool correct = (resultData[i] == expected);
        if (correct) correctCount++;
        
        std::cout << "    " << std::setw(5) << i << " | "
                  << std::setw(5) << inputData[i] << " | "
                  << std::setw(6) << resultData[i] << " | "
                  << std::setw(8) << expected << " | "
                  << (correct ? "✓" : "✗") << std::endl;
    }
    
    std::cout << "\n  Results Summary:" << std::endl;
    std::cout << "    First 10 elements: " << correctCount << "/10 correct" << std::endl;
    
    // Verify all results
    int totalCorrect = 0;
    for (size_t i = 0; i < dataSize; ++i) {
        if (resultData[i] == inputData[i] * 5) {
            totalCorrect++;
        }
    }
    std::cout << "    Total elements:    " << totalCorrect << "/" << dataSize << " correct" << std::endl;
    
    if (totalCorrect == static_cast<int>(dataSize)) {
        std::cout << "  ✓ All results are correct!\n" << std::endl;
    } else {
        std::cout << "  ✗ Some results are incorrect\n" << std::endl;
    }
    
    // Step 8: Cleanup
    std::cout << "Step 8: Cleaning up resources..." << std::endl;
    hostAPI.cuMemFree(inputPtr);
    hostAPI.cuMemFree(resultPtr);
    std::cout << "  ✓ Released device memory\n" << std::endl;
    
    std::cout << "==================================================" << std::endl;
    std::cout << "Demo completed successfully!" << std::endl;
    std::cout << "==================================================" << std::endl;
    
    return 0;
}