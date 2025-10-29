#include "performance_counters.hpp"
#include <iostream>
#include <iomanip>

// Initialize the static counter names map
const std::unordered_map<CounterID, std::string> COUNTER_NAMES = {
    {PerformanceCounterIDs::INVALID, "Invalid Counter"},
    
    // Execution statistics
    {PerformanceCounterIDs::INSTRUCTIONS_EXECUTED, "Instructions Executed"},
    {PerformanceCounterIDs::CYCLES, "Cycles"},
    {PerformanceCounterIDs::IPC, "Instructions Per Cycle"},
    
    // Register statistics
    {PerformanceCounterIDs::REGISTER_READS, "Register Reads"},
    {PerformanceCounterIDs::REGISTER_WRITES, "Register Writes"},
    {PerformanceCounterIDs::SPILL_OPERATIONS, "Spill Operations"},
    
    // Memory statistics
    {PerformanceCounterIDs::GLOBAL_MEMORY_READS, "Global Memory Reads"},
    {PerformanceCounterIDs::GLOBAL_MEMORY_WRITES, "Global Memory Writes"},
    {PerformanceCounterIDs::SHARED_MEMORY_READS, "Shared Memory Reads"},
    {PerformanceCounterIDs::SHARED_MEMORY_WRITES, "Shared Memory Writes"},
    {PerformanceCounterIDs::LOCAL_MEMORY_READS, "Local Memory Reads"},
    {PerformanceCounterIDs::LOCAL_MEMORY_WRITES, "Local Memory Writes"},
    
    // Control flow statistics
    {PerformanceCounterIDs::BRANCHES, "Branches"},
    {PerformanceCounterIDs::DIVERGENT_BRANCHES, "Divergent Branches"},
    {PerformanceCounterIDs::BRANCH_RECONVERGENCE_POINTS, "Branch Reconvergence Points"},
    {PerformanceCounterIDs::PREDICATE_SKIPPED, "Predicate Skipped Instructions"},
    
    // Cache statistics
    {PerformanceCounterIDs::INSTRUCTION_CACHE_HITS, "Instruction Cache Hits"},
    {PerformanceCounterIDs::INSTRUCTION_CACHE_MISSES, "Instruction Cache Misses"},
    {PerformanceCounterIDs::DATA_CACHE_HITS, "Data Cache Hits"},
    {PerformanceCounterIDs::DATA_CACHE_MISSES, "Data Cache Misses"}
};

// Private implementation class
class PerformanceCounters::Impl {
public:
    Impl() {
        // Initialize all counters to zero
        reset();
    }
    
    ~Impl() = default;
    
    // Reset all counters to zero
    void reset() {
        for (size_t i = 0; i < PerformanceCounterIDs::MAX_COUNTER_ID; ++i) {
            m_counterValues[i] = 0;
        }
    }
    
    // Increment a counter by a specific value
    void increment(CounterID counterID, CounterValue value) {
        if (counterID < PerformanceCounterIDs::MAX_COUNTER_ID) {
            m_counterValues[counterID] += value;
        }
    }
    
    // Get the value of a counter
    CounterValue getCounterValue(CounterID counterID) const {
        if (counterID < PerformanceCounterIDs::MAX_COUNTER_ID) {
            return m_counterValues[counterID];
        }
        return 0;
    }
    
private:
    // Counter values
    CounterValue m_counterValues[PerformanceCounterIDs::MAX_COUNTER_ID];
};

PerformanceCounters::PerformanceCounters() : pImpl(std::make_unique<Impl>()) {}

PerformanceCounters::~PerformanceCounters() = default;

void PerformanceCounters::reset() {
    pImpl->reset();
}

void PerformanceCounters::increment(CounterID counterID, CounterValue value) {
    pImpl->increment(counterID, value);
}

CounterValue PerformanceCounters::getCounterValue(CounterID counterID) const {
    return pImpl->getCounterValue(counterID);
}

const std::string& PerformanceCounters::getCounterName(CounterID counterID) const {
    static const std::string unknown("Unknown Counter");
    auto it = COUNTER_NAMES.find(counterID);
    if (it != COUNTER_NAMES.end()) {
        return it->second;
    }
    return unknown;
}

void PerformanceCounters::printCounters() const {
    std::cout << "Performance Counters:" << std::endl;
    std::cout << "---------------------" << std::endl;
    
    for (const auto& entry : COUNTER_NAMES) {
        CounterID counterID = entry.first;
        if (counterID != PerformanceCounterIDs::INVALID) {  // Skip invalid counter
            const std::string& name = entry.second;
            CounterValue value = getCounterValue(counterID);
            
            // Skip counters with zero value
            if (value > 0) {
                std::cout << std::left << std::setw(30) << name << ": " << value << std::endl;
            }
        }
    }
    
    // Calculate and print IPC if cycles > 0
    CounterValue instructions = getCounterValue(PerformanceCounterIDs::INSTRUCTIONS_EXECUTED);
    CounterValue cycles = getCounterValue(PerformanceCounterIDs::CYCLES);
    if (cycles > 0) {
        double ipc = static_cast<double>(instructions) / cycles;
        std::cout << std::left << std::setw(30) << "Instructions Per Cycle (IPC)" << ": " << ipc << std::endl;
    }
}

// Convenience methods for debugger
CounterValue PerformanceCounters::getTotalInstructions() const {
    return getCounterValue(PerformanceCounterIDs::INSTRUCTIONS_EXECUTED);
}

CounterValue PerformanceCounters::getArithmeticInstructions() const {
    // For now, we'll return a sum of different instruction types
    // This is a simplification - in a real implementation you might track arithmetic instructions separately
    return getCounterValue(PerformanceCounterIDs::INSTRUCTIONS_EXECUTED) - 
           getCounterValue(PerformanceCounterIDs::BRANCHES) -
           getCounterValue(PerformanceCounterIDs::GLOBAL_MEMORY_READS) -
           getCounterValue(PerformanceCounterIDs::GLOBAL_MEMORY_WRITES) -
           getCounterValue(PerformanceCounterIDs::SHARED_MEMORY_READS) -
           getCounterValue(PerformanceCounterIDs::SHARED_MEMORY_WRITES);
}

CounterValue PerformanceCounters::getMemoryInstructions() const {
    return getCounterValue(PerformanceCounterIDs::GLOBAL_MEMORY_READS) +
           getCounterValue(PerformanceCounterIDs::GLOBAL_MEMORY_WRITES) +
           getCounterValue(PerformanceCounterIDs::SHARED_MEMORY_READS) +
           getCounterValue(PerformanceCounterIDs::SHARED_MEMORY_WRITES) +
           getCounterValue(PerformanceCounterIDs::LOCAL_MEMORY_READS) +
           getCounterValue(PerformanceCounterIDs::LOCAL_MEMORY_WRITES);
}

CounterValue PerformanceCounters::getControlFlowInstructions() const {
    return getCounterValue(PerformanceCounterIDs::BRANCHES);
}

CounterValue PerformanceCounters::getExecutionTime() const {
    return getCounterValue(PerformanceCounterIDs::CYCLES);
}
