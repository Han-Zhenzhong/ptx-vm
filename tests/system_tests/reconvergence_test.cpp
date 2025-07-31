#include <gtest/gtest.h>
#include "src/core/vm.hpp"
#include "src/execution/reconvergence_mechanism.hpp"

// Test fixture for reconvergence mechanism tests
class ReconvergenceTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize VM and components
        m_vm = std::make_unique<PTXVM>();
        ASSERT_TRUE(m_vm->initialize());
        
        // Get reference to executor
        m_executor = &m_vm->getExecutor();
        
        // Initialize reconvergence mechanism
        m_reconvergence = std::make_unique<ReconvergenceMechanism>();
        ASSERT_TRUE(m_reconvergence->initialize(RECONVERGENCE_ALGORITHM_CFG_BASED));
    }

    void TearDown() override {
        // Clean up
        m_vm.reset();
        m_executor = nullptr;
        m_reconvergence.reset();
    }

    // Helper function to create a simple control flow graph
    void buildSimpleCFG(std::vector<std::vector<size_t>>& cfg) {
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
        //cfg[2].push_back(3);
    }

    // Helper function to simulate branch execution
    void simulateBranchExecution(ReconvergenceMechanism& reconvergence, size_t& currentPC, uint64_t& activeMask, uint64_t threadMask) {
        // Simulate branch instruction
        DecodedInstruction instr;
        instr.type = InstructionTypes::BRA;
        instr.hasPredicate = true;
        instr.predicateValue = (threadMask != 0);
        instr.sources.push_back({OperandType::IMMEDIATE, 1});  // Jump to PC 1
        
        // Handle the branch
        bool takeBranch = (threadMask != 0);
        size_t nextPC = 0;
        
        reconvergence.handleBranch(instr, nextPC, activeMask, threadMask);
        
        // Update PC
        currentPC = nextPC;
    }

    std::unique_ptr<PTXVM> m_vm;
    PTXExecutor* m_executor = nullptr;
    std::unique_ptr<ReconvergenceMechanism> m_reconvergence;
};

// Test basic reconvergence mechanism initialization
TEST_F(ReconvergenceTest, BasicInitialization) {
    // Check that reconvergence mechanism was initialized
    const DivergenceStats& stats = m_reconvergence->getDivergenceStats();
    EXPECT_EQ(stats.numDivergentPaths, 0);
    EXPECT_EQ(stats.maxDivergenceDepth, 0);
    
    // Print stats
    m_reconvergence->printStats();
}

// Test basic divergence handling
TEST_F(ReconvergenceTest, BasicDivergenceHandling) {
    // Set up a simple control flow
    std::vector<std::vector<size_t>> cfg;
    buildSimpleCFG(cfg);
    m_reconvergence->setControlFlowGraph(cfg);
    
    // Initialize execution state
    size_t currentPC = 0;
    uint64_t activeMask = 0xFFFFFFFF;  // All threads active
    uint64_t threadMask1 = 0xFFFF0000;  // Half threads take branch
    uint64_t threadMask2 = 0x0000FFFF;  // Other half take other branch
    
    // First check - handle divergence for first thread mask
    m_reconvergence->handleBranch(DecodedInstruction(), currentPC, activeMask, threadMask1);
    
    // Verify state after first divergence
    EXPECT_EQ(m_reconvergence->getDivergenceStackDepth(), 1);
    EXPECT_EQ(activeMask, threadMask1);
    
    // Move to next instruction
    currentPC = 1;
    
    // Second check - handle divergence for second thread mask
    m_reconvergence->handleBranch(DecodedInstruction(), currentPC, activeMask, threadMask2);
    
    // Verify state after second divergence
    EXPECT_EQ(m_reconvergence->getDivergenceStackDepth(), 2);
    
    // Reset for next test
    m_reconvergence->reset();
}

// Test CFG-based reconvergence point detection
TEST_F(ReconvergenceTest, CFGBasedReconvergence) {
    // Set up a simple control flow
    std::vector<std::vector<size_t>> cfg;
    buildSimpleCFG(cfg);
    m_reconvergence->setControlFlowGraph(cfg);
    
    // Test finding optimal reconvergence points
    size_t reconvergencePoint = m_reconvergence->findOptimalReconvergencePoint(0);
    
    // The optimal reconvergence point should be after both paths
    // In our simple case, this is instruction index 2
    EXPECT_EQ(reconvergencePoint, 2);
    
    // Test reconvergence statistics
    const DivergenceStats& stats = m_reconvergence->getDivergenceStats();
    EXPECT_EQ(stats.averageDivergenceRate, 0.0);
    
    // Reset for next test
    m_reconvergence->reset();
}

// Test stack-based predication
TEST_F(ReconvergenceTest, StackBasedPredication) {
    // Initialize with stack-based algorithm
    m_reconvergence->initialize(RECONVERGENCE_ALGORITHM_STACK_BASED);
    
    // Set up a simple control flow
    std::vector<std::vector<size_t>> cfg;
    buildSimpleCFG(cfg);
    m_reconvergence->setControlFlowGraph(cfg);
    
    // Initial execution state
    size_t currentPC = 0;
    uint64_t activeMask = 0xFFFFFFFF;  // All threads active
    uint64_t threadMask1 = 0xFFFF0000;  // Half threads take branch
    
    // Take the first branch
    simulateBranchExecution(*m_reconvergence, currentPC, activeMask, threadMask1);
    
    // We should have one entry on the divergence stack
    EXPECT_EQ(m_reconvergence->getDivergenceStackDepth(), 1);
    
    // Now reach the reconvergence point
    currentPC = 2;  // Join point
    bool reconverged = m_reconvergence->checkReconvergence(currentPC, activeMask);
    
    // Should have detected reconvergence
    EXPECT_TRUE(reconverged);
    
    // Active mask should be restored to full mask
    EXPECT_EQ(activeMask, 0xFFFFFFFF);
    
    // Divergence stack should be empty now
    EXPECT_EQ(m_reconvergence->getDivergenceStackDepth(), 0);
    
    // Reset for next test
    m_reconvergence->reset();
}

// Test divergence stack depth tracking
TEST_F(ReconvergenceTest, DivergenceStackDepth) {
    // Test maximum divergence depth
    for (size_t i = 0; i < 10; ++i) {
        // Simulate nested divergence
        size_t currentPC = i;
        uint64_t activeMask = 0xFFFFFFFF;
        uint64_t threadMask = (i % 2 == 0) ? 0xFFFF0000 : 0x0000FFFF;
        
        // Handle branch
        simulateBranchExecution(*m_reconvergence, currentPC, activeMask, threadMask);
        
        // Check stack depth
        EXPECT_EQ(m_reconvergence->getDivergenceStackDepth(), i + 1);
    }
    
    // Reset for next test
    m_reconvergence->reset();
}

// Test average divergence rate calculation
TEST_F(ReconvergenceTest, AverageDivergenceRate) {
    // Simulate multiple branches with varying divergence
    for (size_t i = 0; i < 100; ++i) {
        // Alternate between divergent and non-divergent branches
        size_t currentPC = i;
        uint64_t activeMask = 0xFFFFFFFF;
        uint64_t threadMask = (i % 3 == 0) ? 0xFFFFFFFF : 0xFFFF0000;  // Some pattern of divergence
        
        // Handle branch
        simulateBranchExecution(*m_reconvergence, currentPC, activeMask, threadMask);
    }
    
    // Check divergence statistics
    const DivergenceStats& stats = m_reconvergence->getDivergenceStats();
    
    // There were 100 branches, approximately 66% of them should be divergent
    // Actual value will vary based on simulation logic
    EXPECT_GT(stats.averageDivergenceRate, 50.0);  // More than 50% divergence
    
    // Reset for next test
    m_reconvergence->reset();
}

// Test divergence impact factor
TEST_F(ReconvergenceTest, DivergenceImpactFactor) {
    // Create deep divergence scenario
    for (size_t i = 0; i < 20; ++i) {
        // Simulate nested divergence
        size_t currentPC = i;
        uint64_t activeMask = 0xFFFFFFFF;
        uint64_t threadMask = (i % 2 == 0) ? 0xFFFF0000 : 0x0000FFFF;
        
        // Handle branch
        simulateBranchExecution(*m_reconvergence, currentPC, activeMask, threadMask);
    }
    
    // Check that we have a significant impact factor
    const DivergenceStats& stats = m_reconvergence->getDivergenceStats();
    
    // With deep divergence, impact factor should be > 1.0
    // (averageReconvergenceTime * maxDivergenceDepth) / 100.0
    EXPECT_GT(stats.divergenceImpactFactor, 1.0);
    
    // Reset for next test
    m_reconvergence->reset();
}

// Test different reconvergence algorithms
TEST_F(ReconvergenceTest, AlgorithmSelection) {
    // Test basic algorithm
    m_reconvergence->initialize(RECONVERGENCE_ALGORITHM_BASIC);
    
    // Simulate some divergence
    size_t currentPC = 0;
    uint64_t activeMask = 0xFFFFFFFF;
    uint64_t threadMask = 0xFFFF0000;
    
    // Handle branch
    simulateBranchExecution(*m_reconvergence, currentPC, activeMask, threadMask);
    
    // For basic algorithm, reconvergence should happen at next instruction
    currentPC = 1;
    bool reconverged = m_reconvergence->checkReconvergence(currentPC, activeMask);
    
    EXPECT_TRUE(reconverged);
    EXPECT_EQ(activeMask, 0xFFFFFFFF);
    
    // Reset and test CFG-based algorithm
    m_reconvergence->reset();
    m_reconvergence->initialize(RECONVERGENCE_ALGORITHM_CFG_BASED);
    
    // Build simple CFG
    std::vector<std::vector<size_t>> cfg;
    buildSimpleCFG(cfg);
    m_reconvergence->setControlFlowGraph(cfg);
    
    // Handle branch
    currentPC = 0;
    activeMask = 0xFFFFFFFF;
    threadMask = 0xFFFF0000;
    
    simulateBranchExecution(*m_reconvergence, currentPC, activeMask, threadMask);
    
    // CFG-based algorithm should use proper reconvergence point
    currentPC = 2;  // Join point in our simple CFG
    reconverged = m_reconvergence->checkReconvergence(currentPC, activeMask);
    
    EXPECT_TRUE(reconverged);
    EXPECT_EQ(activeMask, 0xFFFFFFFF);
    
    // Reset and test stack-based algorithm
    m_reconvergence->reset();
    m_reconvergence->initialize(RECONVERGENCE_ALGORITHM_STACK_BASED);
    
    // Handle branch
    currentPC = 0;
    activeMask = 0xFFFFFFFF;
    threadMask = 0xFFFF0000;
    
    simulateBranchExecution(*m_reconvergence, currentPC, activeMask, threadMask);
    
    // Stack-based algorithm requires explicit reconvergence points
    currentPC = 1;
    reconverged = m_reconvergence->checkReconvergence(currentPC, activeMask);
    
    EXPECT_FALSE(reconverged);  // Not at reconvergence point yet
    
    // Reach actual reconvergence point
    currentPC = 2;
    reconverged = m_reconvergence->checkReconvergence(currentPC, activeMask);
    
    EXPECT_TRUE(reconverged);
    EXPECT_EQ(activeMask, 0xFFFFFFFF);
    
    // Reset for next test
    m_reconvergence->reset();
}