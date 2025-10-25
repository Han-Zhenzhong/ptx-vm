#ifndef WARP_SCHEDULER_HPP
#define WARP_SCHEDULER_HPP

#include <cstdint>
#include <vector>
#include <deque>
#include <memory>
#include "instruction_types.hpp"
#include "registers/register_bank.hpp"
#include "memory/memory.hpp"

class WarpScheduler;

class Warp {
public:
    // Constructor/destructor
    Warp(uint32_t warpId, uint32_t numThreads);
    ~Warp();

    // Get warp ID
    uint32_t getWarpId() const { return m_warpId; }

    // Get number of threads in this warp
    uint32_t getNumThreads() const { return m_numThreads; }

    // Check if all threads are active
    bool allActive() const;

    // Check if any thread is active
    bool anyActive() const;

    // Get mask of active threads
    uint64_t getActiveMask() const { return m_activeMask; }

    // Set active mask for threads
    void setActiveMask(uint64_t mask) { m_activeMask = mask; }

    // Get current program counter for this warp
    size_t getCurrentPC() const { return m_currentPC; }

    // Set current program counter for this warp
    void setCurrentPC(size_t pc) { m_currentPC = pc; }

    // Get next program counter for this warp
    size_t getNextPC() const { return m_nextPC; }

    // Set next program counter for this warp
    void setNextPC(size_t pc) { m_nextPC = pc; }

    // Get the thread's program counter
    size_t getThreadPC(uint32_t threadId) const;

    // Set the thread's program counter
    void setThreadPC(uint32_t threadId, size_t pc);

    // Get divergence stack depth for this warp
    uint32_t getDivergenceStackDepth() const { return m_divergenceStackDepth; }

    // Push divergence point to stack
    void pushDivergencePoint(size_t joinPC);

    // Pop divergence point from stack
    size_t popDivergencePoint();

    // Check if divergence stack is empty
    bool isDivergenceStackEmpty() const;

private:
    // Structure to represent a divergence point
    struct DivergencePoint {
        size_t joinPC;          // PC where threads should reconverge
        uint64_t activeMask;    // Mask of threads that need to reconverge
    };

    uint32_t m_warpId;                // ID of this warp
    uint32_t m_numThreads;            // Number of threads in this warp
    uint64_t m_activeMask;            // Bitmask of active threads
    size_t m_currentPC;               // Current PC for this warp
    size_t m_nextPC;                  // Next PC for this warp
    uint32_t m_divergenceStackDepth;  // Current depth of divergence stack
    std::vector<size_t> m_divergenceStack;  // Stack of divergence points
    std::vector<size_t> m_threadPCs;        // Program counters for each thread
};

// Structure to track instruction issue information
typedef struct InstructionIssueInfo {
    uint32_t warpId;         // Which warp does this belong to
    size_t instructionIndex;  // Index in instruction stream
    uint64_t activeMask;      // Active threads executing this instruction
} InstructionIssueInfo;

// Execution state for a single thread
typedef struct ThreadExecutionState {
    size_t pc;                // Program counter
    bool active;              // Is this thread active?
    bool completed;           // Has this thread completed execution?
} ThreadExecutionState;

// SIMT execution mode
typedef enum {
    WARP_SCHEDULER_STRICT_RECONVERGENT,  // Strictly reconvergent scheduling
    WARP_SCHEDULER_FOUR_WIDE,             // Four-wide wavefront scheduling
    WARP_SCHEDULER_SERIALIZE_ALL,         // Serialize all divergent branches
    WARP_SCHEDULER_SERIALIZE_SYNC_ONLY    // Serialize only when sync is needed
} SimtExecutionMode;

// Sync state structure
struct SyncState {
    uint32_t ctaId = 0;
    uint32_t gridId = 0;
    size_t syncPC = 0;
    uint64_t arrivalMask = 0;
    size_t syncCount = 0;
};

// Constants for synchronization
const size_t WARPS_PER_CTA = 16;  // Simplified assumption for demonstration
const size_t CTAS_PER_GRID = 32;  // Simplified assumption for demonstration

class WarpScheduler {
public:
    // Constructor/destructor
    explicit WarpScheduler(uint32_t numWarps, uint32_t threadsPerWarp = 32);
    ~WarpScheduler();

    // Initialize the scheduler
    bool initialize();

    // Reset the scheduler for new kernel launch
    void reset();

    // Get number of warps
    uint32_t getNumWarps() const;

    // Get number of threads per warp
    uint32_t getThreadsPerWarp() const;

    // Get current warp being executed
    uint32_t getCurrentWarp() const;

    // Get active threads in a warp
    uint64_t getActiveThreads(uint32_t warpId) const;

    // Set active threads in a warp
    void setActiveThreads(uint32_t warpId, uint64_t activeMask);

    // Get current PC for a warp
    size_t getCurrentPC(uint32_t warpId) const;

    // Set current PC for a warp
    void setCurrentPC(uint32_t warpId, size_t pc);

    // Get next PC for a warp
    size_t getNextPC(uint32_t warpId) const;

    // Set next PC for a warp
    void setNextPC(uint32_t warpId, size_t pc);

    // Select next warp to execute
    uint32_t selectNextWarp();

    // Issue instruction from current warp
    bool issueInstruction(InstructionIssueInfo& issueInfo);

    // Complete execution of an instruction
    void completeInstruction(const InstructionIssueInfo& issueInfo);

    // Handle branch divergence
    void handleBranchDivergence(uint32_t warpId, 
                               uint64_t takenMask,
                               size_t targetPC,
                               size_t fallthroughPC);

    // Check if warp has work
    bool warpHasWork(uint32_t warpId) const;

    // Check if all warps have completed execution
    bool allWarpsComplete() const;

    // Synchronization primitives
    bool syncThreadsInCta(uint32_t ctaId, size_t syncPC);
    bool syncThreadsInGrid(uint32_t gridId, size_t syncPC);
    
    // Synchronization state management
    bool checkCtaThreadsCompleted(uint32_t ctaId);
    bool checkGridCtasCompleted(uint32_t gridId);
    
private:
    // Private implementation details
    class Impl;
    std::unique_ptr<Impl> pImpl;
};

#endif // WARP_SCHEDULER_HPP