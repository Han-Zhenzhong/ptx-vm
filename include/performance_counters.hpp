#ifndef PERFORMANCE_COUNTERS_HPP
#define PERFORMANCE_COUNTERS_HPP

#include <cstdint>
#include <unordered_map>
#include <string>
#include <memory>
#include "execution/reconvergence_mechanism.hpp"

// Performance counter IDs
typedef uint32_t CounterID;

namespace PerformanceCounterIDs {
    enum : CounterID {
        INVALID = 0,
        
        // Execution statistics
        INSTRUCTIONS_EXECUTED,      // Total instructions executed
        CYCLES,                     // Total cycles simulated
        IPC,                        // Instructions per cycle
        
        // Register statistics
        REGISTER_READS,             // Total register reads
        REGISTER_WRITES,            // Total register writes
        SPILL_OPERATIONS,           // Register spill operations
        
        // Memory statistics
        GLOBAL_MEMORY_READS,        // Global memory reads
        GLOBAL_MEMORY_WRITES,       // Global memory writes
        SHARED_MEMORY_READS,        // Shared memory reads
        SHARED_MEMORY_WRITES,       // Shared memory writes
        LOCAL_MEMORY_READS,         // Local memory reads
        LOCAL_MEMORY_WRITES,        // Local memory writes
        PARAMETER_MEMORY_READS,     // Parameter memory reads
        PARAMETER_MEMORY_WRITES,    // Parameter memory writes
        CONSTANT_MEMORY_READS,      // Constant memory reads
        TEXTURE_MEMORY_READS,       // Texture memory reads
        
        // Control flow statistics
        BRANCHES,                   // Total branches executed
        DIVERGENT_BRANCHES,         // Number of divergent branches
        BRANCH_RECONVERGENCE_POINTS, // Branch reconvergence points
        PREDICATE_SKIPPED,          // Predicate skipped instructions
        
        // Cache statistics
        INSTRUCTION_CACHE_HITS,     // Instruction cache hits
        INSTRUCTION_CACHE_MISSES,   // Instruction cache misses
        DATA_CACHE_HITS,            // Data cache hits
        DATA_CACHE_MISSES,          // Data cache misses
        
        // Maximum counter ID
        MAX_COUNTER_ID
    };
} // namespace PerformanceCounterIDs

// Performance counter value type
typedef uint64_t CounterValue;

// Performance counter names
extern const std::unordered_map<CounterID, std::string> COUNTER_NAMES;

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
    
    // Convenience methods for debugger
    CounterValue getTotalInstructions() const;
    CounterValue getArithmeticInstructions() const;
    CounterValue getMemoryInstructions() const;
    CounterValue getControlFlowInstructions() const;
    CounterValue getExecutionTime() const;
    
private:
    // Private implementation details
    class Impl;
    std::unique_ptr<Impl> pImpl;
};

#endif // PERFORMANCE_COUNTERS_HPP