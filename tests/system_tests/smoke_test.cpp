#include <gtest/gtest.h>
#include "vm.hpp"
#include "parser/parser.hpp"
#include "memory/memory.hpp"
#include "optimizer/register_allocator.hpp"
#include "performance_counters.hpp"
#include <cstring>

// Test fixture for system-level tests
class SystemSmokeTest : public ::testing::Test {
protected:
    std::unique_ptr<PTXVM> vm;
    
    void SetUp() override {
        // Create and initialize the VM
        vm = std::make_unique<PTXVM>();
        ASSERT_TRUE(vm->initialize());
    }
    
    void TearDown() override {
        // Clean up
        vm.reset();
    }
};

// Test basic program execution
TEST_F(SystemSmokeTest, TestBasicProgramExecution) {
    // Load the program
    bool loaded = vm->loadProgram("examples/simple_math_example.ptx");
    ASSERT_TRUE(loaded) << "Failed to load program";
    EXPECT_TRUE(vm->isProgramLoaded());
    
    // Allocate memory for results (20 bytes for 5 int32 values)
    CUdeviceptr resultPtr = vm->allocateMemory(20);
    ASSERT_NE(resultPtr, 0) << "Failed to allocate memory";
    
    // Set up kernel parameters
    std::vector<KernelParameter> params;
    params.push_back({resultPtr, sizeof(uint64_t), 0});
    vm->setKernelParameters(params);
    
    // Execute the program
    bool executed = vm->run();
    EXPECT_TRUE(executed) << "Program execution failed";
    
    // Get references to core components for verification
    PerformanceCounters& counters = vm->getPerformanceCounters();
    
    // Verify that instructions were executed
    size_t instructionsExecuted = counters.getCounterValue(PerformanceCounterIDs::INSTRUCTIONS_EXECUTED);
    EXPECT_GT(instructionsExecuted, 0u) << "No instructions were executed";
    
    // Verify cycle count
    size_t cycles = counters.getCounterValue(PerformanceCounterIDs::CYCLES);
    EXPECT_GT(cycles, 0u) << "No cycles recorded";
    
    // Verify register operations occurred
    size_t registerReads = counters.getCounterValue(PerformanceCounterIDs::REGISTER_READS);
    size_t registerWrites = counters.getCounterValue(PerformanceCounterIDs::REGISTER_WRITES);
    EXPECT_GT(registerReads, 0u) << "No register reads";
    EXPECT_GT(registerWrites, 0u) << "No register writes";
    
    // Verify memory operations
    size_t globalMemoryWrites = counters.getCounterValue(PerformanceCounterIDs::GLOBAL_MEMORY_WRITES);
    EXPECT_GT(globalMemoryWrites, 0u) << "No global memory writes";
    
    // Read back results
    int32_t results[5];
    bool copied = vm->copyMemoryDtoH(results, resultPtr, sizeof(results));
    ASSERT_TRUE(copied) << "Failed to copy results from device";
    
    // Verify arithmetic results (from simple_math_example.ptx)
    EXPECT_EQ(results[0], 49);   // 42 + 7
    EXPECT_EQ(results[1], 35);   // 42 - 7
    EXPECT_EQ(results[2], 294);  // 42 * 7
    EXPECT_EQ(results[3], 6);    // 42 / 7
    EXPECT_EQ(results[4], 0);    // 42 % 7
    
    // Clean up
    vm->freeMemory(resultPtr);
}

// Test control flow execution
TEST_F(SystemSmokeTest, TestControlFlowExecution) {
    // Load the program
    bool loaded = vm->loadProgram("examples/control_flow_example.ptx");
    ASSERT_TRUE(loaded) << "Failed to load control flow program";
    
    // Allocate memory for input and output
    CUdeviceptr inputPtr = vm->allocateMemory(4);   // 1 int32
    CUdeviceptr resultPtr = vm->allocateMemory(4);  // 1 int32
    ASSERT_NE(inputPtr, 0);
    ASSERT_NE(resultPtr, 0);
    
    // Initialize input (value to be summed 5 times)
    int32_t inputValue = 10;
    bool copied = vm->copyMemoryHtoD(inputPtr, &inputValue, sizeof(inputValue));
    ASSERT_TRUE(copied);
    
    // Set up kernel parameters
    std::vector<KernelParameter> params;
    params.push_back({inputPtr, sizeof(uint64_t), 0});
    params.push_back({resultPtr, sizeof(uint64_t), 8});
    vm->setKernelParameters(params);
    
    // Execute the program
    bool executed = vm->run();
    EXPECT_TRUE(executed) << "Control flow program execution failed";
    
    // Get performance counters
    PerformanceCounters& counters = vm->getPerformanceCounters();
    
    // Check branch statistics
    size_t branches = counters.getCounterValue(PerformanceCounterIDs::BRANCHES);
    EXPECT_GT(branches, 0u) << "No branches executed";
    
    // Check that instructions were executed
    size_t instructionsExecuted = counters.getCounterValue(PerformanceCounterIDs::INSTRUCTIONS_EXECUTED);
    EXPECT_GT(instructionsExecuted, 5u) << "Too few instructions executed";
    
    // Read back result
    int32_t result;
    copied = vm->copyMemoryDtoH(&result, resultPtr, sizeof(result));
    ASSERT_TRUE(copied);
    
    // The program should sum the input 5 times: 10 * 5 = 50
    EXPECT_EQ(result, 50) << "Control flow result incorrect";
    
    // Clean up
    vm->freeMemory(inputPtr);
    vm->freeMemory(resultPtr);
}

// Test memory operations
TEST_F(SystemSmokeTest, TestMemoryOperations) {
    // Load the program
    bool loaded = vm->loadProgram("examples/memory_ops_example.ptx");
    ASSERT_TRUE(loaded) << "Failed to load memory operations program";
    
    // Get reference to memory subsystem
    MemorySubsystem& memory = vm->getMemorySubsystem();
    
    // Allocate memory for test data
    CUdeviceptr srcPtr = vm->allocateMemory(16);   // 4 int32 values
    CUdeviceptr dstPtr = vm->allocateMemory(16);
    ASSERT_NE(srcPtr, 0);
    ASSERT_NE(dstPtr, 0);
    
    // Initialize source data
    int32_t srcData[4] = {10, 20, 30, 40};
    bool copied = vm->copyMemoryHtoD(srcPtr, srcData, sizeof(srcData));
    ASSERT_TRUE(copied);
    
    // Set up kernel parameters (if the program requires them)
    std::vector<KernelParameter> params;
    params.push_back({srcPtr, sizeof(uint64_t), 0});
    params.push_back({dstPtr, sizeof(uint64_t), 8});
    vm->setKernelParameters(params);
    
    // Execute the program
    bool executed = vm->run();
    EXPECT_TRUE(executed) << "Memory operations program execution failed";
    
    // Verify memory operations occurred
    PerformanceCounters& counters = vm->getPerformanceCounters();
    size_t globalReads = counters.getCounterValue(PerformanceCounterIDs::GLOBAL_MEMORY_READS);
    size_t globalWrites = counters.getCounterValue(PerformanceCounterIDs::GLOBAL_MEMORY_WRITES);
    
    EXPECT_GT(globalReads, 0u) << "No global memory reads";
    EXPECT_GT(globalWrites, 0u) << "No global memory writes";
    
    // Read back destination data
    int32_t dstData[4];
    copied = vm->copyMemoryDtoH(dstData, dstPtr, sizeof(dstData));
    ASSERT_TRUE(copied);
    
    // Verify data was processed correctly (implementation-dependent)
    // At minimum, verify that data was written
    bool dataWritten = false;
    for (int i = 0; i < 4; i++) {
        if (dstData[i] != 0) {
            dataWritten = true;
            break;
        }
    }
    EXPECT_TRUE(dataWritten) << "No data written to destination";
    
    // Clean up
    vm->freeMemory(srcPtr);
    vm->freeMemory(dstPtr);
}

// Test overall system integration
TEST_F(SystemSmokeTest, TestSystemIntegration) {
    // Load the program
    bool loaded = vm->loadProgram("examples/simple_math_example.ptx");
    ASSERT_TRUE(loaded);
    
    // Allocate memory
    CUdeviceptr resultPtr = vm->allocateMemory(20);
    ASSERT_NE(resultPtr, 0);
    
    // Set up parameters
    std::vector<KernelParameter> params;
    params.push_back({resultPtr, sizeof(uint64_t), 0});
    vm->setKernelParameters(params);
    
    // Execute
    bool executed = vm->run();
    ASSERT_TRUE(executed);
    
    // Get all component references
    PerformanceCounters& counters = vm->getPerformanceCounters();
    RegisterAllocator& allocator = vm->getRegisterAllocator();
    RegisterBank& registerBank = vm->getRegisterBank();
    PTXExecutor& executor = vm->getExecutor();
    
    // Print out key metrics
    size_t totalInstructions = counters.getCounterValue(PerformanceCounterIDs::INSTRUCTIONS_EXECUTED);
    size_t totalCycles = counters.getCounterValue(PerformanceCounterIDs::CYCLES);
    
    std::cout << "\n=== System Integration Test Results ===" << std::endl;
    std::cout << "Total instructions executed: " << totalInstructions << std::endl;
    std::cout << "Total execution cycles: " << totalCycles << std::endl;
    
    if (totalCycles > 0) {
        double ipc = static_cast<double>(totalInstructions) / totalCycles;
        std::cout << "Instructions per cycle (IPC): " << ipc << std::endl;
    }
    
    // Verify core functionality
    EXPECT_GT(totalInstructions, 0u) << "No instructions executed";
    EXPECT_GT(totalCycles, 0u) << "No cycles recorded";
    
    // Verify register operations
    size_t regReads = counters.getCounterValue(PerformanceCounterIDs::REGISTER_READS);
    size_t regWrites = counters.getCounterValue(PerformanceCounterIDs::REGISTER_WRITES);
    EXPECT_GT(regReads, 0u) << "No register reads";
    EXPECT_GT(regWrites, 0u) << "No register writes";
    
    // Verify memory operations
    size_t memReads = counters.getCounterValue(PerformanceCounterIDs::PARAMETER_MEMORY_READS);
    size_t memWrites = counters.getCounterValue(PerformanceCounterIDs::GLOBAL_MEMORY_WRITES);
    EXPECT_GT(memReads + memWrites, 0u) << "No memory operations";
    
    // Verify execution completed
    EXPECT_TRUE(executor.isExecutionComplete()) << "Execution not marked complete";
    
    // Clean up
    vm->freeMemory(resultPtr);
}

// Test parser and program structure
TEST_F(SystemSmokeTest, TestProgramParsing) {
    // Load the program
    bool loaded = vm->loadProgram("examples/simple_math_example.ptx");
    ASSERT_TRUE(loaded) << "Failed to load program";
    
    // Get executor and verify program structure
    PTXExecutor& executor = vm->getExecutor();
    EXPECT_TRUE(executor.hasProgramStructure()) << "Program structure not available";
    
    // Get the parsed program
    const PTXProgram& program = executor.getProgram();
    
    // Verify metadata
    EXPECT_FALSE(program.metadata.version.empty()) << "PTX version not parsed";
    EXPECT_FALSE(program.metadata.target.empty()) << "PTX target not parsed";
    std::cout << "PTX Version: " << program.metadata.version << std::endl;
    std::cout << "PTX Target: " << program.metadata.target << std::endl;
    
    // Verify we have instructions
    EXPECT_GT(program.instructions.size(), 0u) << "No instructions parsed";
    std::cout << "Total instructions: " << program.instructions.size() << std::endl;
    
    // Verify we have at least one entry function
    EXPECT_FALSE(program.symbolTable.functions.empty()) << "No functions parsed";
    
    // Find the entry kernel
    bool hasEntryKernel = false;
    for (const auto& [name, func] : program.symbolTable.functions) {
        if (func.isEntry) {
            hasEntryKernel = true;
            std::cout << "Entry kernel: " << name << std::endl;
            std::cout << "  Parameters: " << func.parameters.size() << std::endl;
            std::cout << "  Register declarations: " << func.registerDeclarations.size() << std::endl;
            break;
        }
    }
    EXPECT_TRUE(hasEntryKernel) << "No entry kernel found";
}

// Test register allocation and usage
TEST_F(SystemSmokeTest, TestRegisterOperations) {
    // Get reference to register allocator and bank
    RegisterAllocator& allocator = vm->getRegisterAllocator();
    RegisterBank& registerBank = vm->getRegisterBank();
    
    // Test basic register read/write
    uint32_t testRegister = 0;
    uint64_t testValue = 0xDEADBEEF;
    
    // Write to register
    registerBank.writeRegister(testRegister, testValue);
    
    // Read back
    uint64_t readValue = registerBank.readRegister(testRegister);
    EXPECT_EQ(readValue, testValue) << "Register read/write mismatch";
    
    // Test multiple register writes
    for (uint32_t i = 0; i < 10; i++) {
        registerBank.writeRegister(i, i * 100);
    }
    
    // Verify all values
    for (uint32_t i = 0; i < 10; i++) {
        uint64_t value = registerBank.readRegister(i);
        EXPECT_EQ(value, i * 100) << "Register " << i << " has incorrect value";
    }
    
    // Load and run a program to verify registers work with execution
    bool loaded = vm->loadProgram("examples/simple_math_example.ptx");
    ASSERT_TRUE(loaded);
    
    CUdeviceptr resultPtr = vm->allocateMemory(20);
    ASSERT_NE(resultPtr, 0);
    
    std::vector<KernelParameter> params;
    params.push_back({resultPtr, sizeof(uint64_t), 0});
    vm->setKernelParameters(params);
    
    bool executed = vm->run();
    ASSERT_TRUE(executed);
    
    // Verify register operations occurred during execution
    PerformanceCounters& counters = vm->getPerformanceCounters();
    size_t regReads = counters.getCounterValue(PerformanceCounterIDs::REGISTER_READS);
    size_t regWrites = counters.getCounterValue(PerformanceCounterIDs::REGISTER_WRITES);
    
    EXPECT_GT(regReads, 0u) << "No register reads during execution";
    EXPECT_GT(regWrites, 0u) << "No register writes during execution";
    
    std::cout << "Register reads during execution: " << regReads << std::endl;
    std::cout << "Register writes during execution: " << regWrites << std::endl;
    
    // Clean up
    vm->freeMemory(resultPtr);
}

// Test performance counter functionality
TEST_F(SystemSmokeTest, TestPerformanceCounters) {
    PerformanceCounters& counters = vm->getPerformanceCounters();
    
    // Reset counters
    counters.reset();
    
    // Verify all counters are zero after reset
    EXPECT_EQ(counters.getCounterValue(PerformanceCounterIDs::INSTRUCTIONS_EXECUTED), 0u);
    EXPECT_EQ(counters.getCounterValue(PerformanceCounterIDs::CYCLES), 0u);
    EXPECT_EQ(counters.getCounterValue(PerformanceCounterIDs::REGISTER_READS), 0u);
    
    // Load and execute a program
    bool loaded = vm->loadProgram("examples/simple_math_example.ptx");
    ASSERT_TRUE(loaded);
    
    CUdeviceptr resultPtr = vm->allocateMemory(20);
    ASSERT_NE(resultPtr, 0);
    
    std::vector<KernelParameter> params;
    params.push_back({resultPtr, sizeof(uint64_t), 0});
    vm->setKernelParameters(params);
    
    bool executed = vm->run();
    ASSERT_TRUE(executed);
    
    // Verify counters have been updated
    size_t instructions = counters.getCounterValue(PerformanceCounterIDs::INSTRUCTIONS_EXECUTED);
    size_t cycles = counters.getCounterValue(PerformanceCounterIDs::CYCLES);
    
    EXPECT_GT(instructions, 0u) << "Instruction counter not updated";
    EXPECT_GT(cycles, 0u) << "Cycle counter not updated";
    
    // Print all counters
    std::cout << "\n=== Performance Counter Summary ===" << std::endl;
    counters.printCounters();
    
    // Clean up
    vm->freeMemory(resultPtr);
}

// Test memory allocation and deallocation
TEST_F(SystemSmokeTest, TestMemoryManagement) {
    // Test single allocation
    CUdeviceptr ptr1 = vm->allocateMemory(1024);
    EXPECT_NE(ptr1, 0u) << "Failed to allocate memory";
    
    // Test multiple allocations
    CUdeviceptr ptr2 = vm->allocateMemory(2048);
    CUdeviceptr ptr3 = vm->allocateMemory(512);
    EXPECT_NE(ptr2, 0u);
    EXPECT_NE(ptr3, 0u);
    EXPECT_NE(ptr1, ptr2) << "Allocated same address twice";
    EXPECT_NE(ptr2, ptr3) << "Allocated same address twice";
    
    // Test memory copy to device
    int32_t hostData[256];
    for (int i = 0; i < 256; i++) {
        hostData[i] = i;
    }
    
    bool copied = vm->copyMemoryHtoD(ptr1, hostData, sizeof(hostData));
    EXPECT_TRUE(copied) << "Failed to copy to device";
    
    // Test memory copy from device
    int32_t readBack[256];
    std::memset(readBack, 0, sizeof(readBack));
    
    copied = vm->copyMemoryDtoH(readBack, ptr1, sizeof(readBack));
    EXPECT_TRUE(copied) << "Failed to copy from device";
    
    // Verify data integrity
    bool dataMatches = true;
    for (int i = 0; i < 256; i++) {
        if (readBack[i] != hostData[i]) {
            dataMatches = false;
            break;
        }
    }
    EXPECT_TRUE(dataMatches) << "Memory data corruption detected";
    
    // Test deallocation
    bool freed1 = vm->freeMemory(ptr1);
    bool freed2 = vm->freeMemory(ptr2);
    bool freed3 = vm->freeMemory(ptr3);
    
    EXPECT_TRUE(freed1) << "Failed to free memory";
    EXPECT_TRUE(freed2) << "Failed to free memory";
    EXPECT_TRUE(freed3) << "Failed to free memory";
    
    // Verify allocations are cleaned up
    const auto& allocations = vm->getMemoryAllocations();
    EXPECT_EQ(allocations.size(), 0u) << "Memory leaks detected";
}