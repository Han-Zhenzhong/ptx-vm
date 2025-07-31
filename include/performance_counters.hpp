#ifndef PERFORMANCE_COUNTERS_HPP
#define PERFORMANCE_COUNTERS_HPP

#include <cstdint>
#include <unordered_map>
#include <string>

// Performance counter IDs
typedef uint32_t CounterID;

namespace PerformanceCounters {
    enum : CounterID {
        INVALID = 0,
        
        // Execution statistics
        INSTRUCTIONS_EXECUTED,      // Total instructions executed
        CYCLES,                     // Total cycles simulated
        IPC,                        // Instructions per cycle
        
        // Register statistics
        REGISTER_READS,             // Total register reads
        REGISTER_WRITES,            // Total register writes
        
        // Memory statistics
        GLOBAL_MEMORY_READS,        // Global memory reads
        GLOBAL_MEMORY_WRITES,       // Global memory writes
        SHARED_MEMORY_READS,        // Shared memory reads
        SHARED_MEMORY_WRITES,       // Shared memory writes
        LOCAL_MEMORY_READS,         // Local memory reads
        LOCAL_MEMORY_WRITES,        // Local memory writes
        
        // Control flow statistics
        BRANCHES,                   // Total branches executed
        DIVERGENT_BRANCHES,         // Number of divergent branches
        BRANCH_RECONVERGENCE_POINTS, // Branch reconvergence points
        
        // Cache statistics
        INSTRUCTION_CACHE_HITS,     // Instruction cache hits
        INSTRUCTION_CACHE_MISSES,   // Instruction cache misses
        DATA_CACHE_HITS,            // Data cache hits
        DATA_CACHE_MISSES,          // Data cache misses
        
        // Maximum counter ID
        MAX_COUNTER_ID
    };
} // namespace PerformanceCounters

// Performance counter value type
typedef uint64_t CounterValue;

// Performance counter names
extern const std::unordered_map<CounterID, std::string> COUNTER_NAMES;

// Divergence statistics
struct DivergenceStats {
    size_t numDivergentPaths;         // Total number of divergent paths
    size_t maxDivergenceDepth;        // Maximum depth of divergence stack
    double averageDivergenceRate;     // Average rate of divergence
    double averageReconvergenceTime;  // Average time to reconverge
    double divergenceImpactFactor;    // Impact on performance
};

// Branch statistics
struct BranchStats {
    size_t totalBranches;
    size_t unconditionalBranches;
    size_t divergentBranches;
    size_t errors;
};

// Performance counters
struct PerformanceCounters {
    // Execution statistics
    CounterValue instructionsExecuted;
    CounterValue cycles;
    double ipc;
    
    // Register statistics
    CounterValue registerReads;
    CounterValue registerWrites;
    
    // Memory statistics
    CounterValue globalMemoryReads;
    CounterValue globalMemoryWrites;
    CounterValue sharedMemoryReads;
    CounterValue sharedMemoryWrites;
    CounterValue localMemoryReads;
    CounterValue localMemoryWrites;
    
    // Control flow statistics
    CounterValue branches;
    CounterValue divergentBranches;
    CounterValue branchReconvergencePoints;
    
    // Cache statistics
    CounterValue instructionCacheHits;
    CounterValue instructionCacheMisses;
    CounterValue dataCacheHits;
    CounterValue dataCacheMisses;
    
    // Divergence handling statistics
    DivergenceStats divergenceStats;
    
    // Branch statistics
    BranchStats branchStats;
    
    // Initialize divergence statistics
    void initDivergenceStats();
    
    // Reset divergence statistics
    void resetDivergenceStats();
    
    // Initialize branch statistics
    void initBranchStats();
    
    // Reset branch statistics
    void resetBranchStats();
};

// Performance counters interface class
class PerformanceCounters {
public:
    PerformanceCounters();
    ~PerformanceCounters();
    
    // Reset all counters to zero
    void reset();
    
    // Increment a counter by a specific value
    void increment(CounterID counterID, CounterValue value = 1);
    
    // Get the value of a counter
    CounterValue getCounterValue(CounterID counterID) const;
    
    // Get the name of a counter
    const std::string& getCounterName(CounterID counterID) const;
    
    // Print all counters to stdout
    void printCounters() const;
    
private:
    // Counter values
    CounterValue m_counterValues[PerformanceCounters::MAX_COUNTER_ID];
    
    // Private implementation details
    class Impl;
    std::unique_ptr<Impl> pImpl;
};

#endif // PERFORMANCE_COUNTERS_HPP