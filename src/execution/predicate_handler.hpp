#ifndef PREDICATE_HANDLER_HPP
#define PREDICATE_HANDLER_HPP

#include <cstdint>
#include <stack>
#include <memory>
#include "instruction_types.hpp"

class PredicateHandler;

typedef uint32_t PredicateID;

typedef struct {
    bool value;          // Current predicate value
    bool isNegated;      // Is this a negated predicate (!p)
    bool isValid;        // Is this predicate valid
} PredicateState;

typedef enum {
    EXECUTION_MODE_NORMAL,         // Normal execution (no predication)
    EXECUTION_MODE_PREDICATED,     // Predicated execution
    EXECUTION_MODE_MASKED,         // Masked execution
    EXECUTION_MODE_SIMT            // Full SIMT execution with reconvergence
} ExecutionMode;

typedef enum {
    RECONVERGENCE_STACK_SIZE = 16  // Maximum depth of divergence stack
} ReconvergenceConstants;

#include "reconvergence_mechanism.hpp"

// Add divergence handling support
struct DivergenceStackEntry {
    size_t joinPC;           // Program counter where threads should reconverge
    uint64_t activeMask;     // Active threads mask before divergence
    uint64_t divergentMask;  // Threads that took the branch
    bool isJoinPointValid;   // Whether the join point is valid
};

typedef std::stack<DivergenceStackEntry> DivergenceStack;

class PredicateHandler {
public:
    // Constructor/destructor
    PredicateHandler();
    ~PredicateHandler();

    // Handle divergence and reconvergence in SIMT execution
    void handleDivergenceReconvergence(const DecodedInstruction& instruction, 
                                      size_t& currentPC, 
                                      uint64_t& activeMask);

    // Handle SIMT divergence for instruction execution
    void handleSIMTDivergence(const DecodedInstruction& instruction, 
                             size_t& currentPC, 
                             uint64_t& activeMask,
                             uint64_t threadMask);

    // Initialize the predicate handler
    bool initialize();

    // Set execution mode
    void setExecutionMode(ExecutionMode mode);

    // Get current execution mode
    ExecutionMode getExecutionMode() const;

    // Evaluate predicate for current instruction
    bool evaluatePredicate(const DecodedInstruction& instruction) const;

    // Set predicate state for an instruction
    void setPredicateState(PredicateID predicateId, bool value, bool negated = false);

    // Get predicate state
    const PredicateState* getPredicateState(PredicateID predicateId) const;

    // Check if instruction should execute based on predicate
    bool shouldExecute(const DecodedInstruction& instruction) const;

    // Handle branch instruction with predicate
    void handleBranch(const DecodedInstruction& instruction, size_t& nextPC, uint64_t& activeMask);

    // Handle divergence stack for SIMT execution
    void pushDivergencePoint(size_t joinPC, uint64_t activeMask, uint64_t divergentMask);
    
    // Pop divergence point from stack
    bool popDivergencePoint(size_t& joinPC, uint64_t& activeMask, uint64_t& divergentMask) const;

    // Check if divergence stack is empty
    bool isDivergenceStackEmpty() const;

    // Get current active thread mask
    uint64_t getActiveThreads() const;

    // Set active thread mask
    void setActiveThreads(uint64_t activeMask);

    // Update thread activity after synchronization
    void updateAfterSync(uint64_t activeMask);

    // Get number of active threads
    uint32_t getActiveThreadCount() const;
    
    // Get divergence stack for a warp
    const DivergenceStack& getDivergenceStack(uint32_t warpId) const;
    
    // Set control flow graph for CFG-based reconvergence
    void setControlFlowGraph(const std::vector<std::vector<size_t>>& cfg);
    
private:
    // Private implementation details
    class Impl;
    std::unique_ptr<Impl> pImpl;
    
    // Divergence handling
    DivergenceStack m_divergenceStack;  // Stack to track divergence points
    size_t m_divergenceStartCycle;      // Cycle when divergence started
    
    // Control flow graph for CFG-based reconvergence
    std::vector<std::vector<size_t>>* m_controlFlowGraph = nullptr;
    
    // Divergence statistics
    DivergenceStats stats;
    size_t m_numDivergences = 0;
    
    // Current algorithm
    ReconvergenceAlgorithm m_algorithm = RECONVERGENCE_ALGORITHM_BASIC;
    
    // Algorithm-specific implementations
    void basicReconvergence(const DecodedInstruction& instruction, 
                          size_t& nextPC, 
                          uint64_t& activeMask,
                          bool takeBranch,
                          uint64_t threadMask);
    
    void cfgBasedReconvergence(const DecodedInstruction& instruction, 
                             size_t& nextPC, 
                             uint64_t& activeMask,
                             bool takeBranch,
                             uint64_t threadMask);
    
    void stackBasedReconvergence(const DecodedInstruction& instruction, 
                               size_t& nextPC, 
                               uint64_t& activeMask,
                               bool takeBranch,
                               uint64_t threadMask);
    
    // Find CFG-based reconvergence point
    size_t findCFGReconvergencePoint(const DecodedInstruction& instruction);
    
    // Update divergence statistics
    void updateDivergenceStats(bool takeBranch);
    
    // Update reconvergence statistics
    void updateReconvergenceStats(size_t divergenceCycles);
};

#endif // PREDICATE_HANDLER_HPP