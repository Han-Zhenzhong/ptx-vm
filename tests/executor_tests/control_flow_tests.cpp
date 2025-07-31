#include <gtest/gtest.h>
#include "execution/executor.hpp"
#include "memory/memory.hpp"
#include "registers/register_bank.hpp"

// Test fixture for control flow operations
class ControlFlowTest : public ::testing::Test {
protected:
    std::unique_ptr<PTXExecutor> executor;
    
    void SetUp() override {
        executor = std::make_unique<PTXExecutor>();
        
        // Initialize with empty instructions
        std::vector<PTXInstruction> instructions;
        executor->initialize(instructions);
    }
};

// Helper function to create a register operand
Operand makeRegister(uint32_t regIndex) {
    Operand op;
    op.type = OperandType::REGISTER;
    op.registerIndex = regIndex;
    return op;
}

// Helper function to create an immediate operand
Operand makeImmediate(int64_t value) {
    Operand op;
    op.type = OperandType::IMMEDIATE;
    op.immediateValue = value;
    return op;
}

// Helper function to create a memory operand
Operand makeMemory(uint64_t address) {
    Operand op;
    op.type = OperandType::MEMORY;
    op.address = address;
    return op;
}

// Test BRA instruction (direct branch)
TEST_F(ControlFlowTest, TestBRA_Direct) {
    // Create BRA instruction: bra 5
    DecodedInstruction instr;
    instr.type = InstructionTypes::BRA;
    
    // Source operand: immediate value 5
    Operand target;
    target.type = OperandType::IMMEDIATE;
    target.immediateValue = 5;
    instr.sources.push_back(target);
    
    // Set initial instruction index to 0
    executor->setCurrentInstructionIndexForTest(0);
    
    // Execute the instruction
    EXPECT_TRUE(executor->executeDecodedInstruction(instr));
    
    // Check that we jumped to instruction index 5
    EXPECT_EQ(executor->getCurrentInstructionIndex(), 5);
}

// Test BRA instruction (indirect branch)
TEST_F(ControlFlowTest, TestBRA_Indirect) {
    // Create BRA instruction: bra %r1
    DecodedInstruction instr;
    instr.type = InstructionTypes::BRA;
    
    // Source operand: register %r1
    Operand target;
    target.type = OperandType::REGISTER;
    target.registerIndex = 1;
    instr.sources.push_back(target);
    
    // Set register %r1 to 7
    executor->getRegisterBank().writeRegister(1, 7);
    
    // Set initial instruction index to 0
    executor->setCurrentInstructionIndexForTest(0);
    
    // Execute the instruction
    EXPECT_TRUE(executor->executeDecodedInstruction(instr));
    
    // Check that we jumped to instruction index 7
    EXPECT_EQ(executor->getCurrentInstructionIndex(), 7);
}

// Test simple loop using BRA
TEST_F(ControlFlowTest, TestLoop) {
    // This test creates a small program that counts down from 5 to 0 using a loop
    
    // Our simple program will be at indices 0-3:
    // 0: mov %r1, %r2   // Copy counter to %r1
    // 1: sub %r1, %r1, 1  // Decrement counter
    // 2: st [%r3], %r1   // Store counter to memory
    // 3: bra 1 if %r1 > 0  // Loop if counter > 0
    // 4: exit              // Exit when done
    
    // We'll implement this in our test by executing each instruction individually
    // and checking state at each step
    
    // First initialize memory and registers
    executor->getRegisterBank().writeRegister(2, 5);  // Initialize counter to 5 in %r2
    executor->getRegisterBank().writeRegister(3, 0x1000);  // Address for storing counter
    
    // Create temporary storage for our test
    uint64_t* memory = static_cast<uint64_t*>(executor->getMemorySubsystem().getMemoryBuffer(MemorySpace::GLOBAL));
    ASSERT_NE(memory, nullptr);
    
    // Test execution of sequence
    size_t currentIdx = 0;
    
    // 1. First execute MOV: %r1 = %r2
    {
        DecodedInstruction movInstr;
        movInstr.type = InstructionTypes::MOV;
        movInstr.dest = makeRegister(1);
        movInstr.sources.push_back(makeRegister(2));
        
        EXPECT_TRUE(executor->executeDecodedInstruction(movInstr));
        EXPECT_EQ(executor->getRegisterBank().readRegister(1), 5);  // %r1 should be 5
        
        currentIdx++;
    }
    
    // 2. Then repeatedly execute SUB, ST, and BRA until counter reaches 0
    while (currentIdx < 4) {
        if (currentIdx == 1) {
            // SUB instruction: %r1 = %r1 - 1
            DecodedInstruction subInstr;
            subInstr.type = InstructionTypes::SUB;
            subInstr.dest = makeRegister(1);
            subInstr.sources.push_back(makeRegister(1));
            subInstr.sources.push_back(makeImmediate(1));
            
            EXPECT_TRUE(executor->executeDecodedInstruction(subInstr));
            
            // Check that counter was decremented
            EXPECT_EQ(executor->getRegisterBank().readRegister(1), 5 - (currentIdx - 1));
        } else if (currentIdx == 2) {
            // ST instruction: store %r1 to memory
            DecodedInstruction stInstr;
            stInstr.type = InstructionTypes::ST;
            
            // Memory operand: address in %r3
            Operand memOp;
            memOp.type = OperandType::MEMORY;
            memOp.address = executor->getRegisterBank().readRegister(3);  // %r3 contains address
            
            stInstr.sources.push_back(memOp);
            stInstr.sources.push_back(makeRegister(1));
            
            EXPECT_TRUE(executor->executeDecodedInstruction(stInstr));
            
            // Check that value was stored to memory
            uint64_t* memory = static_cast<uint64_t*>(executor->getMemorySubsystem().getMemoryBuffer(MemorySpace::GLOBAL));
            ASSERT_NE(memory, nullptr);
            
            uint64_t storedValue = *memory;
            EXPECT_EQ(storedValue, executor->getRegisterBank().readRegister(1));
        } else if (currentIdx == 3) {
            // BRA instruction: branch back to instruction 1 if %r1 > 0
            DecodedInstruction braInstr;
            braInstr.type = InstructionTypes::BRA;
            
            // Source operand: immediate value 1 (target instruction)
            Operand target;
            target.type = OperandType::IMMEDIATE;
            target.immediateValue = 1;
            braInstr.sources.push_back(target);
            
            // For simplicity, we simulate a predicate of true
            braInstr.hasPredicate = true;
            braInstr.predicateValue = true;
            
            EXPECT_TRUE(executor->executeDecodedInstruction(braInstr));
            
            // Check that we branched back to instruction 1
            // Note: The actual branch behavior depends on the counter value
            // In this simplified test, we always branch back
            EXPECT_EQ(executor->getCurrentInstructionIndex(), 1);
        }
        
        currentIdx++;
    }
    
    // After loop completes, check final value
    EXPECT_EQ(executor->getRegisterBank().readRegister(1), 0);  // Counter should be 0
}