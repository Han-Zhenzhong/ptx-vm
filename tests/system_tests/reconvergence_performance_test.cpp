#include <gtest/gtest.h>
#include "vm.hpp"
#include "performance_counters.hpp"
#include "src/execution/reconvergence_mechanism.hpp"

// Test fixture for performance tests
class ReconvergencePerformanceTest : public ::testing::Test {
protected:
    std::unique_ptr<VM> vm;
    
    void SetUp() override {
        vm = std::make_unique<VM>();
        ASSERT_TRUE(vm->initialize());
    }
    
    void TearDown() override {
        vm.reset();
    }
};

// Test for basic reconvergence algorithm
TEST_F(ReconvergencePerformanceTest, BasicReconvergence) {
    // Load test program
    ASSERT_TRUE(vm->loadProgram("examples/divergence_test.ptx"));
    
    // Set basic reconvergence algorithm
    vm->setReconvergenceAlgorithm(RECONVERGENCE_ALGORITHM_BASIC);
    
    // Run program
    ASSERT_TRUE(vm->run());
    
    // Get performance counters
    const PerformanceCounters& counters = vm->getPerformanceCounters();
    
    // Verify basic reconvergence metrics
    EXPECT_GT(counters.reconvergence_events, 0);
    EXPECT_LT(counters.divergence_impact, 1000);  // Arbitrary threshold
    EXPECT_GT(counters.instructions_executed, 1000);
}

// Test for CFG-based reconvergence algorithm
TEST_F(ReconvergencePerformanceTest, CFGReconvergence) {
    // Load test program
    ASSERT_TRUE(vm->loadProgram("examples/divergence_test.ptx"));
    
    // Set CFG-based reconvergence algorithm
    vm->setReconvergenceAlgorithm(RECONVERGENCE_ALGORITHM_CFG_BASED);
    
    // Run program
    ASSERT_TRUE(vm->run());
    
    // Get performance counters
    const PerformanceCounters& counters = vm->getPerformanceCounters();
    
    // Verify CFG-based reconvergence metrics
    EXPECT_GT(counters.reconvergence_events, 0);
    EXPECT_LT(counters.divergence_impact, 500);  // Should be better than basic
    EXPECT_GT(counters.instructions_executed, 1000);
    
    // Verify CFG analysis
    EXPECT_GT(counters.cfg_nodes, 0);
    EXPECT_GT(counters.cfg_edges, 0);
}

// Test for stack-based predication
TEST_F(ReconvergencePerformanceTest, StackBasedPredication) {
    // Load test program
    ASSERT_TRUE(vm->loadProgram("examples/divergence_test.ptx"));
    
    // Set stack-based predication algorithm
    vm->setReconvergenceAlgorithm(RECONVERGENCE_ALGORITHM_STACK_BASED);
    
    // Run program
    ASSERT_TRUE(vm->run());
    
    // Get performance counters
    const PerformanceCounters& counters = vm->getPerformanceCounters();
    
    // Verify stack-based predication metrics
    EXPECT_GT(counters.reconvergence_events, 0);
    EXPECT_LT(counters.divergence_impact, 300);  // Should be best
    EXPECT_GT(counters.instructions_executed, 1000);
    
    // Verify stack operations
    EXPECT_GT(counters.predication_stack_pushes, 0);
    EXPECT_GT(counters.predication_stack_pops, 0);
}

// Benchmark different reconvergence algorithms
TEST_F(ReconvergencePerformanceTest, AlgorithmComparison) {
    // Load test program
    ASSERT_TRUE(vm->loadProgram("examples/divergence_test.ptx"));
    
    // Test basic algorithm
    vm->setReconvergenceAlgorithm(RECONVERGENCE_ALGORITHM_BASIC);
    vm->run();
    const PerformanceCounters basicCounters = vm->getPerformanceCounters();
    
    // Test CFG-based algorithm
    vm->setReconvergenceAlgorithm(RECONVERGENCE_ALGORITHM_CFG_BASED);
    vm->run();
    const PerformanceCounters cfgCounters = vm->getPerformanceCounters();
    
    // Test stack-based algorithm
    vm->setReconvergenceAlgorithm(RECONVERGENCE_ALGORITHM_STACK_BASED);
    vm->run();
    const PerformanceCounters stackCounters = vm->getPerformanceCounters();
    
    // Verify algorithm comparison
    EXPECT_LT(stackCounters.divergence_impact, cfgCounters.divergence_impact);
    EXPECT_LT(cfgCounters.divergence_impact, basicCounters.divergence_impact);
    
    // Verify execution efficiency
    EXPECT_GT(stackCounters.ipc, basicCounters.ipc);
    EXPECT_GT(cfgCounters.ipc, basicCounters.ipc);
}

// Test with complex control flow
TEST_F(ReconvergencePerformanceTest, ComplexControlFlow) {
    // Load complex control flow program
    ASSERT_TRUE(vm->loadProgram("examples/complex_control_flow.ptx"));
    
    // Test with different algorithms
    vm->setReconvergenceAlgorithm(RECONVERGENCE_ALGORITHM_BASIC);
    vm->run();
    const PerformanceCounters basicCounters = vm->getPerformanceCounters();
    
    vm->setReconvergenceAlgorithm(RECONVERGENCE_ALGORITHM_CFG_BASED);
    vm->run();
    const PerformanceCounters cfgCounters = vm->getPerformanceCounters();
    
    vm->setReconvergenceAlgorithm(RECONVERGENCE_ALGORITHM_STACK_BASED);
    vm->run();
    const PerformanceCounters stackCounters = vm->getPerformanceCounters();
    
    // Verify algorithm performance on complex control flow
    EXPECT_LT(stackCounters.divergence_impact, cfgCounters.divergence_impact);
    EXPECT_LT(cfgCounters.divergence_impact, basicCounters.divergence_impact);
    
    // Verify execution efficiency
    EXPECT_GT(stackCounters.ipc, basicCounters.ipc);
    EXPECT_GT(cfgCounters.ipc, basicCounters.ipc);
    
    // Verify algorithm-specific metrics
    EXPECT_GT(stackCounters.predication_stack_pushes, 0);
    EXPECT_GT(cfgCounters.cfg_nodes, 0);
}
