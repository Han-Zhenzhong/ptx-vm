#include "vm.hpp"
#include <iostream>
#include <iomanip>

int main() {
    PTXVM vm;
    
    if (!vm.initialize()) {
        std::cerr << "Failed to initialize VM" << std::endl;
        return 1;
    }
    
    std::cout << "Loading program..." << std::endl;
    if (!vm.loadProgram("examples/simple_math_example.ptx")) {
        std::cerr << "Failed to load program" << std::endl;
        return 1;
    }
    
    // Allocate memory for results
    CUdeviceptr resultPtr = vm.allocateMemory(20);
    std::cout << "Allocated result memory at: 0x" << std::hex << resultPtr << std::dec << std::endl;
    
    // Set up kernel parameters
    std::vector<KernelParameter> params;
    params.push_back({resultPtr, sizeof(uint64_t), 0});
    vm.setKernelParameters(params);
    
    // Get executor to inspect program
    PTXExecutor& executor = vm.getExecutor();
    const PTXProgram& program = executor.getProgram();
    
    std::cout << "\nProgram instructions:" << std::endl;
    for (size_t i = 0; i < program.instructions.size() && i < 5; ++i) {
        const auto& instr = program.instructions[i];
        std::cout << "  [" << i << "] type=" << static_cast<int>(instr.type) 
                  << " dataType=" << static_cast<int>(instr.dataType)
                  << " memSpace=" << static_cast<int>(instr.memorySpace)
                  << std::endl;
    }
    
    // Execute
    std::cout << "\nExecuting program..." << std::endl;
    if (!vm.run()) {
        std::cerr << "Execution failed" << std::endl;
        return 1;
    }
    
    // Read back register values
    RegisterBank& regBank = vm.getRegisterBank();
    std::cout << "\nRegister values after execution:" << std::endl;
    for (int i = 0; i < 8; ++i) {
        uint64_t val = regBank.readRegister(i);
        std::cout << "  %r" << i << " = " << std::dec << val 
                  << " (0x" << std::hex << val << std::dec << ")" << std::endl;
    }
    
    // Read back results from memory
    int32_t results[5];
    if (!vm.copyMemoryDtoH(results, resultPtr, sizeof(results))) {
        std::cerr << "Failed to copy results" << std::endl;
        return 1;
    }
    
    std::cout << "\nResults from memory:" << std::endl;
    for (int i = 0; i < 5; ++i) {
        std::cout << "  results[" << i << "] = " << results[i] << std::endl;
    }
    
    vm.freeMemory(resultPtr);
    return 0;
}
