#include "gtest/gtest.h"
#include "reconvergence_mechanism.hpp"
#include "executor.hpp"
#include "vm.hpp"
#include "performance_counters.hpp"

// Divergence performance test fixture
class DivergencePerformanceTest : public ::testing::Test {
protected:
    std::unique_ptr<PTXVM> m_vm;
    std::unique_ptr<Executor> m_executor;
    std::unique_ptr<ReconvergenceMechanism> m_reconvergence;
    
    void SetUp() override {
        // Initialize VM components
        m_vm = std::make_unique<PTXVM>();
        ASSERT_TRUE(m_vm->initialize());
        
        // Initialize executor
        m_executor = std::make_unique<Executor>();
        ASSERT_TRUE(m_executor->initialize());
        
        // Initialize reconvergence mechanism with stack-based algorithm
        m_reconvergence = std::make_unique<ReconvergenceMechanism>();
        ASSERT_TRUE(m_reconvergence->initialize(RECONVERGENCE_ALGORITHM_STACK_BASED));
        
        // Set reconvergence mechanism in executor
        m_executor->setReconvergenceMechanism(m_reconvergence.get());
    }
    
    void TearDown() override {
        // Clean up
        m_vm.reset();
        m_executor.reset();
        m_reconvergence.reset();
    }
};

// Test basic divergence handling
TEST_F(DivergencePerformanceTest, BasicDivergenceHandling) {
    // Create simple control flow graph
    std::vector<std::vector<size_t>> cfg;
    buildSimpleCFG(cfg);
    m_executor->setControlFlowGraph(cfg);
    
    // Initialize test data
    size_t currentPC = 0;
    uint64_t activeMask = 0xFFFFFFFF;  // All threads active
    uint64_t threadMask = 0xFFFF0000;  // Half the threads take the branch
    
    // Handle branch divergence
    m_reconvergence->handleBranch(currentPC, activeMask, threadMask);
    
    // Verify divergence state
    EXPECT_EQ(m_reconvergence->getCurrentDivergenceDepth(), 1);
    EXPECT_GT(m_reconvergence->getDivergenceImpact(), 0);
    
    // Move to reconvergence point
    currentPC = 2;
    bool reconverged = m_reconvergence->checkReconvergence(currentPC, activeMask);
    
    // Verify reconvergence
    EXPECT_TRUE(reconverged);
    EXPECT_EQ(m_reconvergence->getCurrentDivergenceDepth(), 0);
    
    // Get and verify statistics
    DivergenceStats stats = m_reconvergence->getDivergenceStats();
    EXPECT_GT(stats.numDivergentPaths, 0);
    EXPECT_GT(stats.averageReconvergenceTime, 0);
    
    // Print performance metrics
    std::cout << "Divergence impact: " << stats.divergenceImpactFactor << std::endl;
    std::cout << "Divergence rate: " << stats.averageDivergenceRate << "%" << std::endl;
    std::cout << "Reconvergence time: " << stats.averageReconvergenceTime << " cycles" << std::endl;
}

// Test nested divergence handling
TEST_F(DivergencePerformanceTest, NestedDivergenceHandling) {
    // Create complex control flow graph
    std::vector<std::vector<size_t>> cfg;
    buildComplexCFG(cfg, 5);  // Build CFG with 5 branches
    m_executor->setControlFlowGraph(cfg);
    
    // Initialize test data
    size_t currentPC = 0;
    uint64_t activeMask = 0xFFFFFFFF;  // All threads active
    uint64_t threadMask = 0xFFFF0000;  // First divergence
    
    // First branch
    m_reconvergence->handleBranch(currentPC, activeMask, threadMask);
    
    // Verify first divergence
    EXPECT_EQ(m_reconvergence->getCurrentDivergenceDepth(), 1);
    
    // Move to second branch
    currentPC = 3;
    threadMask = 0xFF000000;  // Second divergence
    
    m_reconvergence->handleBranch(currentPC, activeMask, threadMask);
    
    // Verify nested divergence
    EXPECT_EQ(m_reconvergence->getCurrentDivergenceDepth(), 2);
    
    // Get and verify statistics
    DivergenceStats stats = m_reconvergence->getDivergenceStats();
    EXPECT_GT(stats.numDivergentPaths, 1);
    EXPECT_GT(stats.maxDivergenceDepth, 1);
    
    // Print performance metrics
    std::cout << "Max divergence depth: " << stats.maxDivergenceDepth << std::endl;
    std::cout << "Divergence impact: " << stats.divergenceImpactFactor << std::endl;
    std::cout << "Divergence rate: " << stats.averageDivergenceRate << "%" << std::endl;
    std::cout << "Reconvergence time: " << stats.averageReconvergenceTime << " cycles" << std::endl;
}

// Test divergence handling with varying divergence rates
TEST_F(DivergencePerformanceTest, DivergenceRatePerformance) {
    // Create control flow graph with multiple branches
    std::vector<std::vector<size_t>> cfg;
    buildComplexCFG(cfg, 10);  // Build CFG with 10 branches
    m_executor->setControlFlowGraph(cfg);
    
    // Test with different divergence rates
    std::vector<uint64_t> testMasks = {
        0x00000000,  // 0% divergence
        0x0000FFFF,  // 50% divergence
        0x00000001,  // 1 thread diverges
        0xFFFF0000,  // 50% divergence
        0xFFFFFFFF   // 100% divergence (no divergence)
    };
    
    // Run test for each divergence rate
    for (size_t i = 0; i < testMasks.size(); ++i) {
        // Reset for each test
        m_reconvergence->reset();
        m_reconvergence->initialize(RECONVERGENCE_ALGORITHM_STACK_BASED);
        
        // Run test with current mask
        size_t currentPC = 0;
        uint64_t activeMask = 0xFFFFFFFF;
        uint64_t threadMask = testMasks[i];
        
        // Handle branch
        m_reconvergence->handleBranch(currentPC, activeMask, threadMask);
        
        // Verify divergence handling
        if (threadMask == 0 || threadMask == activeMask) {
            // No divergence expected
            EXPECT_EQ(m_reconvergence->getCurrentDivergenceDepth(), 0);
        } else {
            // Divergence expected
            EXPECT_GT(m_reconvergence->getCurrentDivergenceDepth(), 0);
        }
        
        // Get statistics
        DivergenceStats stats = m_reconvergence->getDivergenceStats();
        
        // Print performance metrics
        std::cout << "Test " << i << " (divergence rate: " << (stats.averageDivergenceRate) << "%):" << std::endl;
        std::cout << "  Max divergence depth: " << stats.maxDivergenceDepth << std::endl;
        std::cout << "  Divergence impact: " << stats.divergenceImpactFactor << std::endl;
        std::cout << "  Reconvergence time: " << stats.averageReconvergenceTime << " cycles" << std::endl;
        std::cout << "  Divergence events: " << stats.numDivergentPaths << std::endl;
    }
}

// Test performance comparison between algorithms
TEST_F(DivergencePerformanceTest, AlgorithmComparison) {
    // Test with different algorithms
    std::vector<ReconvergenceAlgorithm> algorithms = {
        RECONVERGENCE_ALGORITHM_BASIC,
        RECONVERGENCE_ALGORITHM_CFG_BASED,
        RECONVERGENCE_ALGORITHM_STACK_BASED
    };
    
    // Test parameters
    size_t currentPC = 0;
    uint64_t activeMask = 0xFFFFFFFF;
    uint64_t threadMask = 0xFFFF0000;
    
    // Run test for each algorithm
    for (size_t i = 0; i < algorithms.size(); ++i) {
        // Reset for each test
        m_reconvergence->reset();
        m_reconvergence->initialize(algorithms[i]);
        
        // Build simple CFG
        std::vector<std::vector<size_t>> cfg;
        buildSimpleCFG(cfg);
        m_executor->setControlFlowGraph(cfg);
        
        // Handle branch
        m_reconvergence->handleBranch(currentPC, activeMask, threadMask);
        
        // Move to reconvergence point
        currentPC = 2;
        bool reconverged = m_reconvergence->checkReconvergence(currentPC, activeMask);
        
        // Verify reconvergence
        EXPECT_TRUE(reconverged);
        
        // Get statistics
        DivergenceStats stats = m_reconvergence->getDivergenceStats();
        
        // Print comparison results
        std::cout << "Algorithm " << i << " performance:" << std::endl;
        std::cout << "  Max divergence depth: " << stats.maxDivergenceDepth << std::endl;
        std::cout << "  Divergence impact: " << stats.divergenceImpactFactor << std::endl;
        std::cout << "  Reconvergence time: " << stats.averageReconvergenceTime << " cycles" << std::endl;
        std::cout << "  Divergence events: " << stats.numDivergentPaths << std::endl;
        std::cout << "  Average divergence rate: " << stats.averageDivergenceRate << "%" << std::endl;
        
        // Verify algorithm-specific expectations
        if (algorithms[i] == RECONVERGENCE_ALGORITHM_BASIC) {
            // Basic algorithm should have lower reconvergence time but higher impact
            EXPECT_LT(stats.averageReconvergenceTime, 10.0);  // Expect fast reconvergence
            EXPECT_GT(stats.divergenceImpactFactor, 0.5);      // Higher impact expected
        } else if (algorithms[i] == RECONVERGENCE_ALGORITHM_CFG_BASED) {
            // CFG-based should have better reconvergence
            EXPECT_GT(stats.averageReconvergenceTime, 5.0);  // May take longer
            EXPECT_LT(stats.divergenceImpactFactor, 0.3);     // Lower impact expected
        } else if (algorithms[i] == RECONVERGENCE_ALGORITHM_STACK_BASED) {
            // Stack-based should handle nested divergence well
            EXPECT_GT(stats.maxDivergenceDepth, 1);          // Should handle nested divergence
            EXPECT_LT(stats.divergenceImpactFactor, 0.2);     // Lowest impact expected
        }
    }
}

// Test divergence handling with long-running code sections
TEST_F(DivergencePerformanceTest, LongRunningDivergence) {
    // Create complex control flow graph
    std::vector<std::vector<size_t>> cfg;
    buildComplexCFG(cfg, 20);  // Build CFG with 20 branches
    m_executor->setControlFlowGraph(cfg);
    
    // Initialize test data
    size_t currentPC = 0;
    uint64_t activeMask = 0xFFFFFFFF;  // All threads active
    uint64_t threadMask = 0xFFFF0000;  // Initial divergence
    
    // Handle branch
    m_reconvergence->handleBranch(currentPC, activeMask, threadMask);
    
    // Simulate execution through multiple divergent paths
    for (size_t i = 0; i < 20; ++i) {
        // Advance PC
        currentPC++;
        
        // Check for reconvergence
        bool reconverged = m_reconvergence->checkReconvergence(currentPC, activeMask);
        
        // If not reconverged, update active mask for next iteration
        if (!reconverged) {
            // For stack-based algorithm, update active mask
            activeMask &= threadMask;
        }
    }
    
    // Get statistics
    DivergenceStats stats = m_reconvergence->getDivergenceStats();
    
    // Verify long-running divergence handling
    EXPECT_GT(stats.numDivergentPaths, 0);
    EXPECT_GT(stats.maxDivergenceDepth, 1);
    EXPECT_GT(stats.averageReconvergenceTime, 0);
    
    // Print performance metrics
    std::cout << "Max divergence depth: " << stats.maxDivergenceDepth << std::endl;
    std::cout << "Divergence impact: " << stats.divergenceImpactFactor << std::endl;
    std::cout << "Reconvergence time: " << stats.averageReconvergenceTime << " cycles" << std::endl;
    std::cout << "Divergence events: " << stats.numDivergentPaths << std::endl;
    std::cout << "Average divergence rate: " << stats.averageDivergenceRate << "%" << std::endl;
}

// Helper function to build a simple control flow graph
void DivergencePerformanceTest::buildSimpleCFG(std::vector<std::vector<size_t>>& cfg) {
    // Build a simple CFG with branches and reconvergence points
    // Structure: branch -> path1 -> join, path2 -> join
    
    // 3 instructions: branch, path1, path2
    cfg.resize(3);
    
    // Branch instruction (0) goes to path1 (1) and path2 (2)
    cfg[0].push_back(1);
    cfg[0].push_back(2);
    
    // Path1 ends at join point (after both paths)
    cfg[1].push_back(2);
    
    // No outgoing edges from path2, it falls through
}

// Helper function to build complex CFG
void DivergencePerformanceTest::buildComplexCFG(std::vector<std::vector<size_t>>& cfg, size_t complexity) {
    // Build a complex CFG with multiple branches and reconvergence points
    size_t numInstructions = complexity * 3;
    cfg.resize(numInstructions);
    
    // Create a chain of branches
    for (size_t i = 0; i < numInstructions - 2; i += 3) {
        // Branch instruction goes to two paths
        cfg[i].push_back(i + 1);
        cfg[i].push_back(i + 2);
        
        // Each path ends at next branch
        if (i + 3 < numInstructions) {
            cfg[i + 1].push_back(i + 3);
            cfg[i + 2].push_back(i + 3);
        } else {
            // Last instructions fall through
            cfg[i + 1].push_back(i + 2);
            cfg[i + 2].clear();  // No outgoing edges
        }
    }
    
    // Last instruction has no outgoing edges
    cfg[numInstructions - 1].clear();
}