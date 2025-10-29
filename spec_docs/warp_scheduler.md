# Warp Scheduler Implementation Details

**Authors**: Han-Zhenzhong, TongyiLingma, GitHub Copilot  
**Last Updated**: 2025-10-29

## Overview
The warp scheduler is a critical component of the PTX Virtual Machine, responsible for managing the execution of threads in a SIMT (Single Instruction Multiple Threads) fashion. This document provides detailed technical information about the warp scheduler implementation.

## Key Concepts

### SIMT Execution Model
The VM implements a SIMT execution model similar to real GPUs:
- Threads are organized into warps (groups of 32 threads)
- All threads in a warp execute the same instruction simultaneously
- Predicates determine which threads execute each instruction
- Branches can cause warps to diverge
- Divergent paths must be reconverged

### Warp States
Each warp can be in one of several states:
- **ACTIVE**: Warp is currently executing
- **WAITING**: Warp is waiting for memory operations
- **BLOCKED**: Warp is blocked waiting for synchronization
- **FINISHED**: Warp has completed execution

### Thread Mask
The thread mask tracks which threads in a warp are active:
```cpp
// Thread mask definition
typedef uint32_t ThreadMask;

// Thread mask operations
inline bool isThreadActive(ThreadMask mask, uint32_t threadId) {
    return (mask & (1 << threadId)) != 0;
}

inline ThreadMask setThreadActive(ThreadMask mask, uint32_t threadId) {
    return mask | (1 << threadId);
}

inline ThreadMask clearThreadActive(ThreadMask mask, uint32_t threadId) {
    return mask & ~(1 << threadId);
}
```

## Implementation Details

### Warp Scheduler Interface
The warp scheduler interface is defined in `warp_scheduler.hpp`:
```cpp
// Warp scheduler interface
class WarpScheduler {
public:
    virtual ~WarpScheduler() = default;
    
    // Initialize the scheduler
    virtual bool initialize() = 0;
    
    // Schedule warps for execution
    virtual void schedule() = 0;
    
    // Get current warp ID
    virtual uint32_t getCurrentWarpId() const = 0;
    
    // Get current PC for a warp
    virtual size_t getCurrentPC(uint32_t warpId) const = 0;
    
    // Set PC for a warp
    virtual void setPC(uint32_t warpId, size_t pc) = 0;
    
    // Get warp state
    virtual WarpState getWarpState(uint32_t warpId) const = 0;
    
    // Set warp state
    virtual void setWarpState(uint32_t warpId, WarpState state) = 0;
    
    // Get active thread mask
    virtual ThreadMask getActiveThreads(uint32_t warpId) const = 0;
    
    // Set active thread mask
    virtual void setActiveThreads(uint32_t warpId, ThreadMask mask) = 0;
    
    // Get divergence stack
    virtual const DivergenceStack& getDivergenceStack(uint32_t warpId) const = 0;
    
    // Push divergence state
    virtual void pushDivergenceState(uint32_t warpId, size_t pc, ThreadMask mask) = 0;
    
    // Pop divergence state
    virtual void popDivergenceState(uint32_t warpId) = 0;
    
    // Get number of active warps
    virtual uint32_t getActiveWarps() const = 0;
    
    // Check if all warps are finished
    virtual bool allWarpsFinished() const = 0;
};
```

### Base Scheduler Implementation
The base scheduler implementation provides common functionality:
```cpp
// Base warp scheduler implementation
#include "warp_scheduler.hpp"

class BaseWarpScheduler : public WarpScheduler {
public:
    BaseWarpScheduler(uint32_t numWarps);
    ~BaseWarpScheduler();
    
    bool initialize() override;
    
    uint32_t getCurrentWarpId() const override;
    
    size_t getCurrentPC(uint32_t warpId) const override;
    
    void setPC(uint32_t warpId, size_t pc) override;
    
    WarpState getWarpState(uint32_t warpId) const override;
    
    void setWarpState(uint32_t warpId, WarpState state) override;
    
    ThreadMask getActiveThreads(uint32_t warpId) const override;
    
    void setActiveThreads(uint32_t warpId, ThreadMask mask) override;
    
    const DivergenceStack& getDivergenceStack(uint32_t warpId) const override;
    
    void pushDivergenceState(uint32_t warpId, size_t pc, ThreadMask mask) override;
    
    void popDivergenceState(uint32_t warpId) override;
    
    uint32_t getActiveWarps() const override;
    
    bool allWarpsFinished() const override;
    
protected:
    uint32_t m_numWarps;         // Number of warps in the system
    uint32_t m_currentWarpId;     // Currently executing warp ID
    
    // Warp state tracking
    struct WarpStateInfo {
        size_t pc;                // Program counter
        ThreadMask activeThreads;  // Active threads mask
        WarpState state;          // Current state
        DivergenceStack divergenceStack;  // Divergence stack
    };
    
    WarpStateInfo* m_warpStates;  // Array of warp states
};
```

### Round-Robin Scheduler
The round-robin scheduler implementation:
```cpp
// Round-robin warp scheduler
#include "warp_scheduler.hpp"

class RoundRobinScheduler : public BaseWarpScheduler {
public:
    RoundRobinScheduler(uint32_t numWarps);
    ~RoundRobinScheduler();
    
    void schedule() override;
    
private:
    uint32_t m_nextWarpId;  // Next warp to schedule
};

// Schedule the next warp in round-robin order
void RoundRobinScheduler::schedule() {
    // Find next active warp
    for (uint32_t i = 0; i < m_numWarps; ++i) {
        uint32_t warpId = m_nextWarpId;
        
        // Advance to next warp
        m_nextWarpId = (m_nextWarpId + 1) % m_numWarps;
        
        // Check if warp is active
        if (getWarpState(warpId) == WarpState::ACTIVE) {
            m_currentWarpId = warpId;
            return;
        }
    }
    
    // No active warps found
    m_currentWarpId = INVALID_WARP_ID;
}
```

### Divergence Handling
The scheduler integrates with the divergence handling mechanism:
```cpp
// Divergence stack implementation
#include "warp_scheduler.hpp"

typedef struct {
    size_t pc;          // Program counter at divergence point
    ThreadMask mask;     // Active threads mask
} DivergenceState;

// Divergence stack operations
void BaseWarpScheduler::pushDivergenceState(uint32_t warpId, size_t pc, ThreadMask mask) {
    DivergenceState state;
    state.pc = pc;
    state.mask = mask;
    
    m_warpStates[warpId].divergenceStack.push(state);
}

void BaseWarpScheduler::popDivergenceState(uint32_t warpId) {
    if (!m_warpStates[warpId].divergenceStack.empty()) {
        m_warpStates[warpId].divergenceStack.pop();
    }
}

const DivergenceStack& BaseWarpScheduler::getDivergenceStack(uint32_t warpId) const {
    return m_warpStates[warpId].divergenceStack;
}
```

### Execution Flow
The warp scheduler controls the overall execution flow:

1. Initialize the scheduler
2. While there are active warps:
   a. Select next warp to execute
   b. Set current warp context
   c. Execute instructions for the warp
   d. Handle divergence and control flow
   e. Update warp state

### Instruction Execution
The executor interacts with the warp scheduler:
```cpp
# Warp Scheduler Implementation

## Overview
This document describes the implementation of the warp scheduler component in the PTX Virtual Machine. The warp scheduler is responsible for managing the SIMT execution model, handling branch divergence, and maintaining the execution state of warps.

## Key Concepts

### SIMT Execution Model
The warp scheduler implements a Single Instruction Multiple Threads (SIMT) execution model:
- Threads are organized into warps (typically 32 threads per warp)
- All threads in a warp execute the same instruction
- Predicates determine which threads actually execute each instruction
- Branch divergence causes warps to split into separate paths

### Divergence Handling
The scheduler manages divergence through:
- Active thread masks
- Divergence stack for tracking reconvergence points
- Multiple reconvergence algorithms (basic, CFG-based, stack-based predication)

### Execution State
The scheduler maintains:
- Current PC for each warp
- Active thread mask
- Divergence stack
- Predicate state

## Implementation Details

### Warp Class
The `Warp` class represents a single warp of threads:
```cpp
Warp::Warp(uint32_t warpId, uint32_t numThreads) :
    m_warpId(warpId), 
    m_numThreads(numThreads),
    m_activeMask(0),
    m_currentPC(0),
    m_nextPC(0),
    m_divergenceStackDepth(0) {
    // Allocate thread program counters
    if (m_numThreads > 0 && m_numThreads <= 64) {
        m_threadPCs.resize(m_numThreads);
        for (uint32_t i = 0; i < m_numThreads; ++i) {
            m_threadPCs[i] = m_currentPC;
        }
    }
}
```

Key methods in the `Warp` class:
```cpp
// Check if all threads are active
bool allActive() const;

// Check if any thread is active
bool anyActive() const;

// Get PC for a specific thread
size_t getThreadPC(uint32_t threadId) const;

// Set PC for a specific thread
void setThreadPC(uint32_t threadId, size_t pc);

// Push divergence point onto stack
void pushDivergencePoint(size_t joinPC);

// Pop divergence point from stack
size_t popDivergencePoint();

// Check if divergence stack is empty
bool isDivergenceStackEmpty() const;
```

### Warp Scheduler
The `WarpScheduler` class manages multiple warps:
```cpp
// Handle branch divergence
void WarpScheduler::handleBranchDivergence(uint32_t warpId, 
                                           uint64_t takenMask,
                                           size_t targetPC,
                                           size_t fallthroughPC) {
    // Implementation details
}

// Complete instruction execution
void WarpScheduler::completeInstruction(const InstructionIssueInfo& issueInfo) {
    // Implementation details
}

// Check if warp has work to do
bool WarpScheduler::warpHasWork(uint32_t warpId) const {
    // Implementation details
}

// Check if all warps are complete
bool WarpScheduler::allWarpsComplete() const {
    // Implementation details
}
```

### Divergence Stack
The divergence stack tracks active divergence points:
```cpp
// Push divergence point
void Warp::pushDivergencePoint(size_t joinPC) {
    m_divergenceStack.push_back(joinPC);
    m_divergenceStackDepth++;
}

// Pop divergence point
size_t Warp::popDivergencePoint() {
    if (m_divergenceStack.empty()) {
        return 0;  // No divergence point to pop
    }
    
    size_t top = m_divergenceStack.back();
    m_divergenceStack.pop_back();
    m_divergenceStackDepth--;
    
    return top;
}
```

### Predicate Handling
The predicate handler manages conditional execution:
```cpp
// Evaluate predicate for a thread
PredicateResult PredicateHandler::evaluatePredicate(uint32_t threadId, const DecodedInstruction& instruction) {
    // Implementation details
    return PREDICATE_RESULT_ACTIVE;
}

// Set predicate value for a thread
void PredicateHandler::setPredicateValue(uint32_t threadId, const PredicateValue& value) {
    if (threadId < m_predicateValues.size()) {
        m_predicateValues[threadId] = value;
    }
}

// Get predicate value for a thread
PredicateValue PredicateHandler::getPredicateValue(uint32_t threadId) const {
    PredicateValue result;
    result.value = 0;
    result.valid = false;
    result.dynamic = false;
    
    if (threadId < m_predicateValues.size()) {
        result = m_predicateValues[threadId];
    }
    
    return result;
}
```

## Integration with Other Components

### Executor Integration
The warp scheduler works closely with the executor:
```cpp
// In executor.cpp
#include "warp_scheduler.hpp"

// During initialization
m_warpScheduler = std::make_unique<WarpScheduler>();
if (!m_warpScheduler->initialize(numWarps, threadsPerWarp)) {
    // Handle error
}

// During execution loop
while (executing) {
    // Get current warp
    for (uint32_t warpId = 0; warpId < m_numWarps; ++warpId) {
        // Check if warp has active threads
        if (m_warpScheduler->warpHasWork(warpId)) {
            // Get current instruction
            size_t currentPC = m_warpScheduler->getCurrentPC(warpId);
            const DecodedInstruction& instr = getCurrentInstruction(currentPC);
            
            // Evaluate predicate
            uint64_t activeMask = m_warpScheduler->getActiveThreads(warpId);
            uint64_t threadMask = 0;
            
            if (instr.hasPredicate) {
                // Evaluate predicate for each thread
                for (uint32_t threadId = 0; threadId < threadsPerWarp; ++threadId) {
                    PredicateResult result = m_predicateHandler.evaluatePredicate(threadId, instr);
                    
                    if (result == PREDICATE_RESULT_ACTIVE) {
                        threadMask |= (1ULL << threadId);
                    }
                }
                
                // Update active mask based on predicate
                activeMask &= threadMask;
            }
            
            // Execute instruction if any threads are active
            if (activeMask != 0) {
                executeInstruction(instr, warpId, activeMask);
            }
            
            // Update warp's PC
            m_warpScheduler->setNextPC(warpId, currentPC + 1);
        }
    }
}
```

### Divergence Handling
The scheduler integrates with the divergence reconvergence mechanism:
```cpp
// Handle branch divergence
void WarpScheduler::handleBranchDivergence(uint32_t warpId, 
                                           uint64_t takenMask,
                                           size_t targetPC,
                                           size_t fallthroughPC) {
    pImpl->handleBranchDivergence(warpId, takenMask, targetPC, fallthroughPC);
}

// In the implementation
void WarpScheduler::handleBranchDivergence(uint32_t warpId, 
                                           uint64_t takenMask,
                                           size_t targetPC,
                                           size_t fallthroughPC) {
    // Save divergence point
    Warp* warp = &m_warps[warpId];
    if (warp) {
        // Push divergence point for the path not taken
        warp->pushDivergencePoint(fallthroughPC);
        
        // Update active mask to only threads that took the branch
        warp->setActiveMask(takenMask);
        
        // Set next PC to target address
        warp->setCurrentPC(targetPC);
    }
}
```

### Control Flow Integration
The scheduler uses control flow information:
```cpp
// Set control flow graph
void Executor::setControlFlowGraph(const std::vector<std::vector<size_t>>& cfg) {
    m_controlFlowGraph = cfg;
}

// During execution
const std::vector<size_t>& Executor::getCurrentControlFlow(size_t currentPC) {
    if (currentPC < m_controlFlowGraph.size()) {
        return m_controlFlowGraph[currentPC];
    }
    
    static std::vector<size_t> empty;
    return empty;
}
```

## Performance Impact
The choice of divergence handling algorithm significantly impacts performance:

| Algorithm Type       | Divergence Overhead | Complexity | Accuracy |
|----------------------|---------------------|------------|----------|
| Basic                | High                | Low        | Low      |
| CFG-Based            | Moderate            | Moderate   | High     |
| Stack-Based Predication | Low               | High       | High     |

## Usage Example
```cpp
// Initialize warp scheduler
WarpScheduler* scheduler = createWarpScheduler();
if (!scheduler->initialize(4, 32)) {  // 4 warps, 32 threads per warp
    // Handle error
}

// Initialize predicate handler
PredicateHandler* predicateHandler = createPredicateHandler();
if (!predicateHandler->initialize(32)) {  // 32 predicate registers
    // Handle error
}

// Main execution loop
while (executing) {
    // Process each warp
    for (uint32_t warpId = 0; warpId < 4; ++warpId) {
        // Check if warp has work
        if (scheduler->warpHasWork(warpId)) {
            // Get current PC
            size_t currentPC = scheduler->getCurrentPC(warpId);
            
            // Get current instruction
            const DecodedInstruction& instr = getCurrentInstruction(currentPC);
            
            // Check for branch instruction
            if (isBranchInstruction(instr)) {
                // Handle branch divergence
                handleBranchDivergence(*scheduler, *predicateHandler, warpId, currentPC, instr);
            } else {
                // Execute instruction
                executeInstruction(instr, warpId);
                
                // Move to next PC
                scheduler->setNextPC(warpId, currentPC + 1);
            }
        }
    }
}

// Print statistics
scheduler->printStats();
predicateHandler->printStats();
```

## Future Improvements
Planned enhancements include:
- More sophisticated scheduling algorithms (round-robin, priority-based, etc.)
- Better divergence tracking and statistics
- Enhanced predicate handling with more complex conditions
- Integration with performance counters for detailed analysis
- Support for dynamic warp creation and destruction
- Enhanced debugging capabilities for divergence analysis
- Visualization tools for warp execution patterns
- Integration with the VM profiler for performance analysis
- Enhanced support for different warp sizes
- Better handling of long-running divergent paths