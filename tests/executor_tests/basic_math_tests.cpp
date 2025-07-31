#include <gtest/gtest.h>
#include "execution/executor.hpp"
#include "memory/memory.hpp"
#include "registers/register_bank.hpp"

// Test fixture for basic math operations
class BasicMathTest : public ::testing::Test {
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

// Test ADD instruction
TEST_F(BasicMathTest, TestADD) {
    // Create ADD instruction: %r1 = %r2 + 3
    DecodedInstruction instr;
    instr.type = InstructionTypes::ADD;
    
    // Destination register %r1
    instr.dest = makeRegister(1);
    
    // Source operands: %r2 and 3
    instr.sources.push_back(makeRegister(2));
    instr.sources.push_back(makeImmediate(3));
    
    // Set up register values
    executor->getRegisterBank().writeRegister(2, 5);  // %r2 = 5
    
    // Execute the instruction
    EXPECT_TRUE(executor->executeDecodedInstruction(instr));
    
    // Check that %r1 contains 8 (5+3)
    EXPECT_EQ(executor->getRegisterBank().readRegister(1), 8);
}

// Test SUB instruction
TEST_F(BasicMathTest, TestSUB) {
    // Create SUB instruction: %r1 = %r2 - 2
    DecodedInstruction instr;
    instr.type = InstructionTypes::SUB;
    
    // Destination register %r1
    instr.dest = makeRegister(1);
    
    // Source operands: %r2 and 2
    instr.sources.push_back(makeRegister(2));
    instr.sources.push_back(makeImmediate(2));
    
    // Set up register values
    executor->getRegisterBank().writeRegister(2, 7);  // %r2 = 7
    
    // Execute the instruction
    EXPECT_TRUE(executor->executeDecodedInstruction(instr));
    
    // Check that %r1 contains 5 (7-2)
    EXPECT_EQ(executor->getRegisterBank().readRegister(1), 5);
}

// Test MUL instruction
TEST_F(BasicMathTest, TestMUL) {
    // Create MUL instruction: %r1 = %r2 * 4
    DecodedInstruction instr;
    instr.type = InstructionTypes::MUL;
    
    // Destination register %r1
    instr.dest = makeRegister(1);
    
    // Source operands: %r2 and 4
    instr.sources.push_back(makeRegister(2));
    instr.sources.push_back(makeImmediate(4));
    
    // Set up register values
    executor->getRegisterBank().writeRegister(2, 6);  // %r2 = 6
    
    // Execute the instruction
    EXPECT_TRUE(executor->executeDecodedInstruction(instr));
    
    // Check that %r1 contains 24 (6*4)
    EXPECT_EQ(executor->getRegisterBank().readRegister(1), 24);
}

// Test DIV instruction
TEST_F(BasicMathTest, TestDIV) {
    // Create DIV instruction: %r1 = %r2 / 3
    DecodedInstruction instr;
    instr.type = InstructionTypes::DIV;
    
    // Destination register %r1
    instr.dest = makeRegister(1);
    
    // Source operands: %r2 and 3
    instr.sources.push_back(makeRegister(2));
    instr.sources.push_back(makeImmediate(3));
    
    // Set up register values
    executor->getRegisterBank().writeRegister(2, 15);  // %r2 = 15
    
    // Execute the instruction
    EXPECT_TRUE(executor->executeDecodedInstruction(instr));
    
    // Check that %r1 contains 5 (15/3)
    EXPECT_EQ(executor->getRegisterBank().readRegister(1), 5);
}

// Test MOV instruction
TEST_F(BasicMathTest, TestMOV) {
    // Create MOV instruction: %r1 = %r2
    DecodedInstruction instr;
    instr.type = InstructionTypes::MOV;
    
    // Destination register %r1
    instr.dest = makeRegister(1);
    
    // Source operand: %r2
    instr.sources.push_back(makeRegister(2));
    
    // Set up register values
    executor->getRegisterBank().writeRegister(2, 42);  // %r2 = 42
    
    // Execute the instruction
    EXPECT_TRUE(executor->executeDecodedInstruction(instr));
    
    // Check that %r1 contains 42
    EXPECT_EQ(executor->getRegisterBank().readRegister(1), 42);
}

// Test complex sequence of math operations
TEST_F(BasicMathTest, TestComplexSequence) {
    // Sequence: %r1 = (%r2 + 5) * (%r3 - 2)
    
    // First execute ADD: %r1 = %r2 + 5
    DecodedInstruction addInstr;
    addInstr.type = InstructionTypes::ADD;
    addInstr.dest = makeRegister(1);
    addInstr.sources.push_back(makeRegister(2));
    addInstr.sources.push_back(makeImmediate(5));
    
    // Then execute MUL: %r1 = %r1 * (%r3 - 2)
    DecodedInstruction mulInstr;
    mulInstr.type = InstructionTypes::MUL;
    mulInstr.dest = makeRegister(1);
    
    // First source is %r1 (result of ADD)
    Operand src0;
    src0.type = OperandType::REGISTER;
    src0.registerIndex = 1;
    
    // Second source is %r3 - 2
    Operand src1;
    src1.type = OperandType::IMMEDIATE;
    src1.immediateValue = 2;
    
    mulInstr.sources.push_back(src0);
    mulInstr.sources.push_back(src1);
    
    // Set up register values
    executor->getRegisterBank().writeRegister(2, 10);  // %r2 = 10
    executor->getRegisterBank().writeRegister(3, 7);   // %r3 = 7
    
    // Execute ADD
    EXPECT_TRUE(executor->executeDecodedInstruction(addInstr));
    
    // Check intermediate result in %r1
    EXPECT_EQ(executor->getRegisterBank().readRegister(1), 15);  // 10+5=15
    
    // Execute MUL
    EXPECT_TRUE(executor->executeDecodedInstruction(mulInstr));
    
    // Check final result in %r1
    EXPECT_EQ(executor->getRegisterBank().readRegister(1), 30);  // 15*2=30
}