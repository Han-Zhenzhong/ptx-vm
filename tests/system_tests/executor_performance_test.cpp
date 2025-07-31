#include <gtest/gtest.h>
#include "src/execution/executor.hpp"
#include "src/decoder/decoder.hpp"
#include "src/memory/memory.hpp"

// Test fixture for executor performance tests
class ExecutorPerformanceTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize components
        ASSERT_TRUE(m_executor.initialize());
        
        // Build a simple control flow graph
        buildSimpleCFG(m_cfg);
        m_executor.setControlFlowGraph(m_cfg);
        
        // Initialize performance counters
        m_testIterations = 1000;
    }

    void TearDown() override {
        // Clean up
        m_executor.reset();
    }

    // Helper function to build a simple control flow graph
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
    }

    // Helper function to create a sample PTX program
    void createSamplePTXProgram(std::vector<DecodedInstruction>& instructions) {
        // Create a simple PTX program with branches
        instructions.clear();
        
        // Instruction 0: Branch instruction
        DecodedInstruction braInstr;
        braInstr.type = InstructionTypes::BRA;
        braInstr.hasPredicate = true;
        braInstr.predicateRegister = 0;
        braInstr.sources.push_back({OperandType::IMMEDIATE, 1});  // Jump to PC 1
        instructions.push_back(braInstr);
        
        // Instruction 1: Simple math operation (path1)
        DecodedInstruction addInstr;
        addInstr.type = InstructionTypes::ADD;
        addInstr.hasPredicate = false;
        addInstr.destination = {OperandType::REGISTER, 0};
        addInstr.sources.push_back({OperandType::REGISTER, 1});
        addInstr.sources.push_back({OperandType::IMMEDIATE, 42});
        instructions.push_back(addInstr);
        
        // Instruction 2: Simple math operation (path2)
        instructions.push_back(addInstr);
        
        // Instruction 3: Join point with another operation
        instructions.push_back(addInstr);
        
        // Set up control flow graph
        m_executor.setControlFlowGraph(m_cfg);
    }

    // Helper function to measure execution performance
    template<typename Func>
    void measurePerformance(Func testFunction, const std::string& testName) {
        // Warm-up phase
        for (size_t i = 0; i < 100; ++i) {
            testFunction();
        }
        
        // Measure performance
        auto start = std::chrono::high_resolution_clock::now();
        
        for (size_t i = 0; i < m_testIterations; ++i) {
            testFunction();
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        
        // Calculate duration
        std::chrono::duration<double> duration = end - start;
        double iterationsPerSecond = m_testIterations / duration.count();
        
        // Print performance metrics
        std::cout << "Performance Metrics for " << testName << ":" << std::endl;
        std::cout << "----------------------------------------" << std::endl;
        std::cout << "Iterations: " << m_testIterations << std::endl;
        std::cout << "Time: " << duration.count() << " seconds" << std::endl;
        std::cout << "Iterations/Second: " << iterationsPerSecond << std::endl;
        std::cout << "Cycles/Iteration: " << (duration.count() * 1e9 / m_testIterations) << " ns" << std::endl;
        std::cout << std::endl;
    }

    // Helper function to simulate execution of a program
    void simulateExecution(const std::vector<DecodedInstruction>& instructions, size_t iterations) {
        // Set instructions
        m_executor.setInstructions(instructions);
        
        // Reset execution state
        m_executor.reset();
        
        // Execute for specified iterations
        for (size_t i = 0; i < iterations; ++i) {
            // Execute instructions
            while (m_executor.getCurrentPC() < instructions.size()) {
                m_executor.executeInstruction();
            }
            
            // Reset PC for next iteration
            m_executor.setCurrentPC(0);
        }
    }

    // Test execution performance with simple math operations
    void testMathPerformance() {
        // Create a program with simple math instructions
        std::vector<DecodedInstruction> instructions;
        instructions.reserve(100);
        
        // Create ADD instruction
        DecodedInstruction addInstr;
        addInstr.type = InstructionTypes::ADD;
        addInstr.hasPredicate = false;
        addInstr.destination = {OperandType::REGISTER, 0};
        addInstr.sources.push_back({OperandType::REGISTER, 1});
        addInstr.sources.push_back({OperandType::IMMEDIATE, 42});
        
        // Create MUL instruction
        DecodedInstruction mulInstr;
        mulInstr.type = InstructionTypes::MUL;
        mulInstr.hasPredicate = false;
        mulInstr.destination = {OperandType::REGISTER, 2};
        mulInstr.sources.push_back({OperandType::REGISTER, 3});
        mulInstr.sources.push_back({OperandType::IMMEDIATE, 3});
        
        // Fill instructions
        for (size_t i = 0; i < 100; ++i) {
            if (i % 2 == 0) {
                instructions.push_back(addInstr);
            } else {
                instructions.push_back(mulInstr);
            }
        }
        
        // Set instructions
        m_executor.setInstructions(instructions);
        
        // Measure performance
        measurePerformance([&]() {
            // Reset PC
            m_executor.setCurrentPC(0);
            
            // Execute all instructions
            while (m_executor.getCurrentPC() < instructions.size()) {
                m_executor.executeInstruction();
            }
        }, "Math Operations");
    }

    // Test execution performance with branches
    void testBranchPerformance() {
        // Create a program with branches
        std::vector<DecodedInstruction> instructions;
        buildBranchingProgram(instructions);
        
        // Set instructions
        m_executor.setInstructions(instructions);
        
        // Measure performance
        measurePerformance([&]() {
            // Reset PC
            m_executor.setCurrentPC(0);
            
            // Execute all instructions
            while (m_executor.getCurrentPC() < instructions.size()) {
                m_executor.executeInstruction();
            }
        }, "Branch Handling");
    }

    // Helper function to build a branching program
    void buildBranchingProgram(std::vector<DecodedInstruction>& instructions) {
        // Create a program with branches and reconvergence
        instructions.clear();
        
        // Instruction 0: Branch instruction
        DecodedInstruction braInstr;
        braInstr.type = InstructionTypes::BRA;
        braInstr.hasPredicate = true;
        braInstr.predicateRegister = 0;
        braInstr.sources.push_back({OperandType::IMMEDIATE, 1});  // Jump to PC 1
        instructions.push_back(braInstr);
        
        // Instruction 1: Simple math operation (path1)
        DecodedInstruction addInstr;
        addInstr.type = InstructionTypes::ADD;
        addInstr.hasPredicate = false;
        addInstr.destination = {OperandType::REGISTER, 0};
        addInstr.sources.push_back({OperandType::REGISTER, 1});
        addInstr.sources.push_back({OperandType::IMMEDIATE, 42});
        instructions.push_back(addInstr);
        
        // Instruction 2: Simple math operation (path2)
        instructions.push_back(addInstr);
        
        // Instruction 3: Join point with another operation
        instructions.push_back(addInstr);
        
        // Set up control flow
        buildSimpleCFG(m_cfg);
        m_executor.setControlFlowGraph(m_cfg);
    }

    // Test execution performance with varying divergence rates
    void testDivergencePerformance(uint64_t threadMask, const std::string& testName) {
        // Create a program with branches
        std::vector<DecodedInstruction> instructions;
        buildBranchingProgram(instructions);
        
        // Set instructions
        m_executor.setInstructions(instructions);
        
        // Reset execution state
        m_executor.reset();
        
        // Measure performance
        measurePerformance([&]() {
            // Reset PC
            m_executor.setCurrentPC(0);
            
            // Execute all instructions
            while (m_executor.getCurrentPC() < instructions.size()) {
                // Get current instruction
                const DecodedInstruction& instr = instructions[m_executor.getCurrentPC()];
                
                // Handle branch divergence
                if (instr.type == InstructionTypes::BRA && instr.hasPredicate) {
                    // Simulate branch with specified thread mask
                    m_executor.handleBranch(instr, m_executor.getCurrentPC(), m_executor.getActiveMask(), threadMask);
                } else {
                    // Execute simple instruction
                    m_executor.executeInstruction();
                }
            }
        }, testName);
    }

    Executor m_executor;
    std::vector<std::vector<size_t>> m_cfg;
    size_t m_testIterations;
};

// Test basic execution performance
TEST_F(ExecutorPerformanceTest, BasicExecutionPerformance) {
    // Create a simple program with math operations
    std::vector<DecodedInstruction> instructions;
    buildBranchingProgram(instructions);
    
    // Measure performance
    measurePerformance([&]() {
        // Reset PC
        m_executor.setCurrentPC(0);
        
        // Execute all instructions
        while (m_executor.getCurrentPC() < instructions.size()) {
            m_executor.executeInstruction();
        }
    }, "Basic Execution");
}

// Test execution performance with math operations
TEST_F(ExecutorPerformanceTest, MathOperationsPerformance) {
    testMathPerformance();
}

// Test execution performance with branches
TEST_F(ExecutorPerformanceTest, BranchHandlingPerformance) {
    testBranchPerformance();
}

// Test execution performance with full divergence
TEST_F(ExecutorPerformanceTest, FullDivergencePerformance) {
    // All threads take different paths
    uint64_t fullDivergenceMask = 0xFFFFFFFF;
    testDivergencePerformance(fullDivergenceMask, "Full Divergence");
}

// Test execution performance with partial divergence
TEST_F(ExecutorPerformanceTest, PartialDivergencePerformance) {
    // Half threads take different paths
    uint64_t partialDivergenceMask = 0xFFFF0000;
    testDivergencePerformance(partialDivergenceMask, "Partial Divergence");
}

// Test execution performance with no divergence
TEST_F(ExecutorPerformanceTest, NoDivergencePerformance) {
    // All threads take same path
    uint64_t noDivergenceMask = 0;
    testDivergencePerformance(noDivergenceMask, "No Divergence");
}

// Test execution performance with nested divergence
TEST_F(ExecutorPerformanceTest, NestedDivergencePerformance) {
    // Reset and initialize
    m_executor.reset();
    
    // Create a complex program with nested branches
    std::vector<DecodedInstruction> instructions;
    instructions.reserve(20);
    
    // Create ADD instruction
    DecodedInstruction addInstr;
    addInstr.type = InstructionTypes::ADD;
    addInstr.hasPredicate = false;
    addInstr.destination = {OperandType::REGISTER, 0};
    addInstr.sources.push_back({OperandType::REGISTER, 1});
    addInstr.sources.push_back({OperandType::IMMEDIATE, 42});
    
    // Create BRA instruction
    DecodedInstruction braInstr;
    braInstr.type = InstructionTypes::BRA;
    braInstr.hasPredicate = true;
    braInstr.predicateRegister = 0;
    braInstr.sources.push_back({OperandType::IMMEDIATE, 1});
    
    // Build a nested branching program
    for (int i = 0; i < 5; ++i) {
        // Outer branch
        instructions.push_back(braInstr);
        
        // Inner branch
        instructions.push_back(braInstr);
        
        // Math operations in each path
        instructions.push_back(addInstr);
        instructions.push_back(addInstr);
    }
    
    // Add join points
    for (int i = 0; i < 5; ++i) {
        instructions.push_back(addInstr);
    }
    
    // Set instructions
    m_executor.setInstructions(instructions);
    
    // Build complex CFG
    std::vector<std::vector<size_t>> nestedCFG;
    buildComplexCFG(nestedCFG, 5);
    m_executor.setControlFlowGraph(nestedCFG);
    
    // Measure performance
    measurePerformance([&]() {
        // Reset PC
        m_executor.setCurrentPC(0);
        
        // Execute all instructions
        while (m_executor.getCurrentPC() < instructions.size()) {
            // Get current instruction
            const DecodedInstruction& instr = instructions[m_executor.getCurrentPC()];
            
            // Handle branch divergence
            if (instr.type == InstructionTypes::BRA && instr.hasPredicate) {
                // Simulate branch with full divergence
                m_executor.handleBranch(instr, m_executor.getCurrentPC(), m_executor.getActiveMask(), 0xFFFFFFFF);
            } else {
                // Execute simple instruction
                m_executor.executeInstruction();
            }
        }
    }, "Nested Divergence");
}

// Helper function to build complex CFG
void ExecutorPerformanceTest::buildComplexCFG(std::vector<std::vector<size_t>>& cfg, size_t complexity) {
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
            cfg[i + 2].push_back(i + 3);
        }
    }
    
    // Last instruction has no outgoing edges
    cfg[numInstructions - 1].clear();
}