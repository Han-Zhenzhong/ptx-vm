# Divergence Handling and Reconvergence Mechanism

**Authors**: Han-Zhenzhong, TongyiLingma, GitHub Copilot  
**Last Updated**: 2025-10-29

## Overview
This document describes the divergence handling and reconvergence mechanism implementation in the PTX Virtual Machine. The mechanism supports multiple algorithms for handling branch divergence and ensuring proper thread reconvergence.

## Key Concepts

### Branch Divergence
Branch divergence occurs when threads within a warp take different paths through conditional branches. This reduces utilization as threads must be executed separately until they reconverge.

### Reconvergence Points
Reconvergence points are locations in the code where divergent threads should meet again. Proper identification and handling of these points is critical for correct execution.

### Divergence Stack
The divergence stack tracks active divergence points, storing information about:
- Join PC (where threads should reconverge)
- Active thread mask before divergence
- Threads that took the branch

## Algorithms Implemented

### 1. Basic Reconvergence Algorithm
- Simplest approach
- Assumes all threads reconverge at next instruction
- Inefficient but easy to implement
- No CFG analysis required

### 2. Control Flow Graph (CFG) Based Reconvergence
- Uses CFG to find natural post-dominator points
- More accurate reconvergence points
- Better performance for complex control flow
- Requires CFG construction during compilation

### 3. Stack-Based Predication
- Most sophisticated algorithm
- Maintains divergence state on a stack
- Handles nested divergence efficiently
- Supports explicit predicated execution
- Most faithful to real GPU behavior

## Implementation Details

### Divergence Handling Process
1. When a branch is encountered, check predicate
2. If branch is taken by some threads but not others:
   - Push entry onto divergence stack
   - Update active mask for divergent path
   - Record start time for statistics
3. Continue execution with modified active mask

### Reconvergence Process
1. At each cycle, check if current PC matches top of stack
2. If match found:
   - Restore full active mask
   - Pop stack
   - Update statistics on divergence duration

### Statistics Collection
The mechanism collects detailed statistics including:
- Number of divergent paths
- Maximum divergence depth
- Average divergence rate
- Average reconvergence time
- Divergence impact factor

## Integration with Other Components

### Executor Integration
The reconvergence mechanism is tightly integrated with the executor:
```cpp
// In executor.cpp
#include "reconvergence_mechanism.hpp"

// During initialization
m_reconvergence = std::make_unique<ReconvergenceMechanism>();
if (!m_reconvergence->initialize(RECONVERGENCE_ALGORITHM_CFG_BASED)) {
    // Handle error
}

// During branch handling
size_t nextPC = m_currentInstructionIndex + 1;
uint64_t activeMask = m_warpScheduler.getActiveThreads(warpId);

if (m_reconvergence->handleBranch(instr, nextPC, activeMask, threadMask)) {
    // Successfully handled divergence
}

// During execution loop
m_reconvergence->updateExecutionState(m_currentInstructionIndex, activeMask);
```

### Control Flow Graph Integration
For CFG-based reconvergence, the executor builds a basic CFG:
```cpp
// Build simple CFG
std::vector<std::vector<size_t>> cfg;
cfg.resize(m_decodedInstructions.size());

for (size_t i = 0; i < m_decodedInstructions.size(); ++i) {
    const DecodedInstruction& instr = m_decodedInstructions[i];
    
    if (instr.type == InstructionTypes::BRA) {
        // For branch instructions, add target to CFG
        if (instr.sources.size() == 1 && 
            instr.sources[0].type == OperandType::IMMEDIATE) {
            size_t target = static_cast<size_t>(instr.sources[0].immediateValue);
            
            if (target < m_decodedInstructions.size()) {
                cfg[i].push_back(target);
                cfg[i].push_back(i + 1);  // Also goes to next instruction
            }
        }
    } else {
        // For non-branch instructions, just go to next instruction
        if (i + 1 < m_decodedInstructions.size()) {
            cfg[i].push_back(i + 1);
        }
    }
}

// Set CFG in reconvergence mechanism
m_reconvergence->setControlFlowGraph(cfg);
```

## Performance Impact
The choice of algorithm significantly impacts performance:

| Algorithm Type       | Divergence Overhead | Complexity | Accuracy |
|----------------------|---------------------|------------|----------|
| Basic                | High                | Low        | Low      |
| CFG-Based            | Moderate            | Moderate   | High     |
| Stack-Based Predication | Low               | High       | High     |

## Usage Example
```cpp
// Initialize reconvergence mechanism
ReconvergenceMechanism* reconvergence = createReconvergenceMechanism();
if (!reconvergence->initialize(RECONVERGENCE_ALGORITHM_STACK_BASED)) {
    // Handle error
}

// Build control flow graph (simplified example)
std::vector<std::vector<size_t>> cfg;
// ... build CFG from decoded instructions ...
reconvergence->setControlFlowGraph(cfg);

// During execution loop
while (executing) {
    // Get current instruction
    const DecodedInstruction& instr = getCurrentInstruction();
    
    // Check if this is a branch instruction
    if (isBranchInstruction(instr)) {
        size_t nextPC = getCurrentPC() + 1;
        uint64_t activeMask = getActiveThreads();
        uint64_t threadMask = getThreadMask();
        
        // Handle the branch and track divergence
        reconvergence->handleBranch(instr, nextPC, activeMask, threadMask);
        
        // Update program counter
        setProgramCounter(nextPC);
    }
    
    // Update execution state
    reconvergence->updateExecutionState(getCurrentPC(), getActiveThreads());
    
    // Execute instruction
    executeInstruction();
}

// Print divergence statistics
reconvergence->printStats();