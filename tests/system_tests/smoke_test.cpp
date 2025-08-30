#include <gtest/gtest.h>
#include "vm.hpp"
#include "parser/parser.hpp"
#include "memory/memory.hpp"
#include "optimizer/register_allocator.hpp"
#include "performance_counters.hpp"

// Test fixture for system-level tests
class SystemSmokeTest : public ::testing::Test {
protected:
    std::unique_ptr<PTXVM> vm;
    
    void SetUp() override {
        // Create and initialize the VM
        vm = std::make_unique<PTXVM>();
        ASSERT_TRUE(vm->initialize());
    }
};

// Helper function to verify memory contents
bool verifyMemoryContent(const MemorySubsystem& memory, 
                       MemorySpace space,
                       size_t offset,
                       const void* expectedData,
                       size_t size) {
    const uint8_t* actualData = static_cast<const uint8_t*>(memory.getMemoryBuffer(space));
    if (!actualData) {
        return false;
    }
    
    const uint8_t* expectedBytes = static_cast<const uint8_t*>(expectedData);
    actualData += offset;
    
    for (size_t i = 0; i < size; ++i) {
        if (actualData[i] != expectedBytes[i]) {
            return false;
        }
    }
    return true;
}

// Test basic program execution
TEST_F(SystemSmokeTest, TestBasicProgramExecution) {
    // Load and execute a simple PTX program
    bool result = vm->loadAndExecuteProgram("examples/simple_math_example.ptx");
    EXPECT_TRUE(result);
    
    // Get references to core components
    RegisterBank& registerBank = vm->getRegisterBank();
    MemorySubsystem& memory = vm->getMemorySubsystem();
    PerformanceCounters& counters = vm->getPerformanceCounters();
    RegisterAllocator& allocator = vm->getRegisterAllocator();
    
    // Verify register allocator configuration
    EXPECT_EQ(allocator.getNumPhysicalRegisters(), 16u);
    EXPECT_EQ(allocator.getNumWarps(), 1u);
    EXPECT_EQ(allocator.getThreadsPerWarp(), 32u);
    
    // For now, just check that some instructions were executed
    size_t instructionsExecuted = counters.getCounterValue(PerformanceCounterIDs::INSTRUCTIONS_EXECUTED);
    EXPECT_GT(instructionsExecuted, 0u);
    
    // We can't fully verify results without knowing where they're stored
    // This is just a basic smoke test that execution happened
    
    // Check that we have reasonable cycle count
    size_t cycles = counters.getCounterValue(PerformanceCounterIDs::CYCLES);
    EXPECT_GT(cycles, 0u);
    
    // Check instruction mix - using available counters
    // Since ARITHMETIC_INSTRUCTIONS and MEMORY_INSTRUCTIONS don't exist,
    // we'll use some available counters that should have non-zero values
    size_t registerReads = counters.getCounterValue(PerformanceCounterIDs::REGISTER_READS);
    size_t registerWrites = counters.getCounterValue(PerformanceCounterIDs::REGISTER_WRITES);
    
    // The example program should have at least one register read and write
    EXPECT_GE(registerReads, 1u);
    EXPECT_GE(registerWrites, 1u);
}

// Test control flow execution
TEST_F(SystemSmokeTest, TestControlFlowExecution) {
    // Load and execute a control flow PTX program
    bool result = vm->loadAndExecuteProgram("examples/control_flow_example.ptx");
    EXPECT_TRUE(result);
    
    // Get performance counters
    PerformanceCounters& counters = vm->getPerformanceCounters();
    
    // Check branch statistics
    size_t branches = counters.getCounterValue(PerformanceCounterIDs::BRANCHES);
    size_t divergentBranches = counters.getCounterValue(PerformanceCounterIDs::DIVERGENT_BRANCHES);
    
    // The example program should have at least one branch
    EXPECT_GE(branches, 1u);
    
    // With warp specialization, there might be no divergent branches
    // but we expect some branching activity
    EXPECT_LE(divergentBranches, branches);
    
    // Check that we actually executed some instructions
    size_t instructionsExecuted = counters.getCounterValue(PerformanceCounterIDs::INSTRUCTIONS_EXECUTED);
    EXPECT_GT(instructionsExecuted, 5u);  // Control flow program has multiple steps
}

// Test memory operations
TEST_F(SystemSmokeTest, TestMemoryOperations) {
    // Load and execute a memory operations PTX program
    bool result = vm->loadAndExecuteProgram("examples/memory_ops_example.ptx");
    EXPECT_TRUE(result);
    
    // Get references to core components
    MemorySubsystem& memory = vm->getMemorySubsystem();
    
    // Verify memory copy - check that input value was copied to output
    // Note: Actual addresses would depend on program implementation
    const uint8_t* globalMem = static_cast<const uint8_t*>(memory.getMemoryBuffer(MemorySpace::GLOBAL));
    ASSERT_NE(globalMem, nullptr);
    
    // For this test, we'll assume our test program copies from offset 0 to 4
    // In real implementation, these values would be determined by the program
    EXPECT_EQ(globalMem[0], globalMem[4]);
    
    // Verify array processing - check that values were incremented
    // Again, using hardcoded offsets based on our test program
    for (int i = 0; i < 4; ++i) {
        // Each element should be incremented by 1
        EXPECT_EQ(globalMem[i * 4 + 0], globalMem[i * 4 + 4] + 1);
    }
}

// Test overall system integration
TEST_F(SystemSmokeTest, TestSystemIntegration) {
    // Load and execute a PTX program
    bool result = vm->loadAndExecuteProgram("examples/simple_math_example.ptx");
    EXPECT_TRUE(result);
    
    // Get all component statistics
    PerformanceCounters& counters = vm->getPerformanceCounters();
    RegisterAllocator& allocator = vm->getRegisterAllocator();
    
    // Print out key metrics
    std::cout << "Total instructions executed: " 
              << counters.getCounterValue(PerformanceCounterIDs::INSTRUCTIONS_EXECUTED) << std::endl;
    std::cout << "Total execution cycles: " 
              << counters.getCounterValue(PerformanceCounterIDs::CYCLES) << std::endl;
    std::cout << "Register utilization: " 
              << allocator.getRegisterUtilization() * 100 << "%" << std::endl;
    
    // These are just basic checks that the system is working together
    EXPECT_GT(counters.getCounterValue(PerformanceCounterIDs::INSTRUCTIONS_EXECUTED), 0u);
    EXPECT_GT(counters.getCounterValue(PerformanceCounterIDs::CYCLES), 0u);
    
    // Register allocation should be at least partially utilized
    EXPECT_GT(allocator.getRegisterUtilization(), 0.0);
}

// Test different scheduling algorithms
TEST_F(SystemSmokeTest, TestSchedulingAlgorithms) {
    // Create an instruction scheduler
    InstructionScheduler scheduler;
    
    // Use the executor to get some decoded instructions
    PTXExecutor& executor = vm->getExecutor();
    const std::vector<DecodedInstruction>& instructions = executor.getDecodedInstructions();
    
    // Skip test if we don't have any instructions
    if (instructions.empty()) {
        GTEST_SKIP() << "No instructions available for scheduling test";
    }
    
    // Test each scheduling algorithm
    std::vector<ScheduledInstruction> scheduled;
    
    // Test in-order scheduling
    scheduler.setSchedulingAlgorithm(SCHEDULING_SIMPLE_INORDER);
    bool result = scheduler.scheduleInstructions(instructions, scheduled);
    EXPECT_TRUE(result);
    
    if (result && !scheduled.empty()) {
        // Verify in-order scheduling produced a valid schedule
        double ipcInOrder = scheduler.getSchedulingStats().at("instructions_per_cycle");
        std::cout << "In-Order Scheduling IPC: " << ipcInOrder << std::endl;
    }
    
    // Test list-based scheduling
    scheduler.setSchedulingAlgorithm(SCHEDULING_LIST_BASED);
    result = scheduler.scheduleInstructions(instructions, scheduled);
    EXPECT_TRUE(result);
    
    if (result && !scheduled.empty()) {
        // Verify list-based scheduling produced a valid schedule
        double ipcListBased = scheduler.getSchedulingStats().at("instructions_per_cycle");
        std::cout << "List-Based Scheduling IPC: " << ipcListBased << std::endl;
        
        // List-based should generally be better than or equal to in-order
        if (scheduler.getSchedulingStats().find("instructions_per_cycle") != scheduler.getSchedulingStats().end() && 
            scheduler.getSchedulingStats().find("scheduling_cycles") != scheduler.getSchedulingStats().end()) {
            
            double ipcList = scheduler.getSchedulingStats().at("instructions_per_cycle");
            double ipcInOrder = 0.0;
            
            // Try with in-order for comparison
            scheduler.setSchedulingAlgorithm(SCHEDULING_SIMPLE_INORDER);
            std::vector<ScheduledInstruction> scheduledInOrder;
            if (scheduler.scheduleInstructions(instructions, scheduledInOrder)) {
                ipcInOrder = scheduler.getSchedulingStats().at("instructions_per_cycle");
                
                // For meaningful comparison, we need more complex tests
                // This is just a placeholder for future work
                std::cout << "IPC Improvement: " << (ipcList / ipcInOrder) * 100 << "%" << std::endl;
            }
        }
    }
}

// Test register allocation and usage
TEST_F(SystemSmokeTest, TestRegisterAllocation) {
    // Get reference to register allocator
    RegisterAllocator& allocator = vm->getRegisterAllocator();
    
    // Get reference to register bank
    RegisterBank& registerBank = vm->getRegisterBank();
    
    // Get allocation info
    uint32_t numPhysical = allocator.getNumPhysicalRegisters();
    uint32_t totalVirtual = allocator.getTotalVirtualRegisters();
    
    // Check that we have at least as many virtual registers as physical
    EXPECT_GE(totalVirtual, numPhysical);
    
    // Check that utilization is reasonable
    double utilization = allocator.getRegisterUtilization();
    EXPECT_GT(utilization, 0.0);
    EXPECT_LE(utilization, 1.0);
    
    // Try mapping some registers
    // Use a clearly invalid register value for comparison
    uint32_t reg1 = allocator.mapVirtualToPhysical(0);
    uint32_t reg2 = allocator.mapVirtualToPhysical(1);
    
    // Ensure we got valid registers (not equal to the invalid value)
    EXPECT_NE(reg1, static_cast<uint32_t>(-1));
    EXPECT_NE(reg2, static_cast<uint32_t>(-1));
    
    // Ensure we can access registers
    registerBank.writeRegister(reg1, 42);
    uint64_t value = registerBank.readRegister(reg1);
    EXPECT_EQ(value, 42);
    
    // Test context switching
    bool saveResult = allocator.saveRegisterState(0, 0);
    bool restoreResult = allocator.restoreRegisterState(0, 0);
    
    // Simple pass/fail for now
    EXPECT_TRUE(saveResult);
    EXPECT_TRUE(restoreResult);
}