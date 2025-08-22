#ifndef DEBUGGER_HPP
#define DEBUGGER_HPP

#include <cstdint>
#include <vector>
#include <unordered_map>
#include <string>
#include <memory>
#include "registers/register_bank.hpp"
#include "memory/memory.hpp"
#include "execution/executor.hpp"

// Forward declaration of implementation class
class DebuggerImpl;

class Debugger {

public:
    // Constructor/destructor
    Debugger(PTXExecutor* executor);
    ~Debugger();

    // Set a breakpoint at the specified instruction index
    bool setBreakpoint(size_t instructionIndex);

    // Clear a breakpoint at the specified instruction index
    bool clearBreakpoint(size_t instructionIndex);

    // Clear all breakpoints
    void clearAllBreakpoints();

    // Check if there is a breakpoint at the specified index
    bool hasBreakpoint(size_t instructionIndex) const;

    // Start execution with debugging
    bool startExecution();

    // Continue execution from current point
    bool continueExecution();

    // Step to next instruction
    bool stepInstruction();

    // Get current instruction index
    size_t getCurrentInstructionIndex() const;

    // Print register state
    void printRegisters() const;

    // Print memory state
    void printMemory(MemorySpace space, uint64_t address, size_t numWords);

    // Disassemble code around current instruction
    void disassembleCurrent(size_t numBefore = 5, size_t numAfter = 5);

    // Print warp execution visualization
    void printWarpVisualization() const;

    // Print memory access visualization
    void printMemoryVisualization() const;

    // Print performance counter display
    void printPerformanceCounters() const;

private:
    // Pimpl idiom - pointer to implementation
    std::unique_ptr<DebuggerImpl> pImpl;
};

#endif // DEBUGGER_HPP