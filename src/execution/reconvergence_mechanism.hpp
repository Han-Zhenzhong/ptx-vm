#ifndef RECONVERGENCE_MECHANISM_HPP
#define RECONVERGENCE_MECHANISM_HPP

#include <cstdint>
#include <vector>
#include <memory>
#include "instruction_types.hpp"

typedef uint32_t PredicateID;

typedef enum {
    RECONVERGENCE_ALGORITHM_BASIC = 0,     // Basic reconvergence algorithm
    RECONVERGENCE_ALGORITHM_CFG_BASED,     // Control Flow Graph-based reconvergence
    RECONVERGENCE_ALGORITHM_STACK_BASED,   // Stack-based predication
    RECONVERGENCE_ALGORITHM_LAST = RECONVERGENCE_ALGORITHM_STACK_BASED
} ReconvergenceAlgorithm;

typedef struct {
    size_t numDivergentPaths;      // Number of paths created by divergence
    size_t maxDivergenceDepth;     // Maximum depth of divergence stack
    double averageDivergenceRate; // Average rate of divergence across all branches
    double averageReconvergenceTime; // Average cycles to reconverge
    double divergenceImpactFactor; // Factor representing impact on performance
} DivergenceStats;

class ReconvergenceMechanism {
public:
    // Constructor/destructor
    ReconvergenceMechanism();
    ~ReconvergenceMechanism();

    // Initialize with specified algorithm
    bool initialize(ReconvergenceAlgorithm algorithm);

    // Reset state
    void reset();

    // Handle a branch instruction and track divergence
    bool handleBranch(const DecodedInstruction& instruction, 
                     size_t instructionIndex,
                     size_t& nextPC, 
                     uint64_t& activeMask,
                     uint64_t threadMask);

    // Update execution state at each cycle
    void updateExecutionState(size_t currentPC, uint64_t activeMask);

    // Get current divergence stack depth
    size_t getDivergenceStackDepth() const;

    // Check if we've reached a reconvergence point
    bool checkReconvergence(size_t currentPC, uint64_t& activeMask);

    // Get divergence statistics
    const DivergenceStats& getDivergenceStats() const;

    // Print divergence statistics
    void printStats() const;

    // Set the control flow graph
    void setControlFlowGraph(const std::vector<std::vector<size_t>>& cfg);

    // Get optimal reconvergence point for an instruction
    size_t findOptimalReconvergencePoint(size_t instructionIndex) const;

private:
    // Private implementation details
    class Impl;
    std::unique_ptr<Impl> pImpl;
};

#endif // RECONVERGENCE_MECHANISM_HPP