#include "performance_counters.hpp"
#include <iostream>
#include <iomanip>

// Initialize the static counter names map
const std::unordered_map<CounterID, std::string> COUNTER_NAMES = {
    {PerformanceCounters::INVALID, "Invalid Counter"},
    
    // Execution statistics
    {PerformanceCounters::INSTRUCTIONS_EXECUTED, "Instructions Executed"},
    {PerformanceCounters::CYCLES, "Cycles"},
    {PerformanceCounters::IPC, "Instructions Per Cycle"},
    
    // Register statistics
    {PerformanceCounters::REGISTER_READS, "Register Reads"},
    {PerformanceCounters::REGISTER_WRITES, "Register Writes"},
    
    // Memory statistics
    {PerformanceCounters::GLOBAL_MEMORY_READS, "Global Memory Reads"},
    {PerformanceCounters::GLOBAL_MEMORY_WRITES, "Global Memory Writes"},
    {PerformanceCounters::SHARED_MEMORY_READS, "Shared Memory Reads"},
    {PerformanceCounters::SHARED_MEMORY_WRITES, "Shared Memory Writes"},
    {PerformanceCounters::LOCAL_MEMORY_READS, "Local Memory Reads"},
    {PerformanceCounters::LOCAL_MEMORY_WRITES, "Local Memory Writes"},
    
    // Control flow statistics
    {PerformanceCounters::BRANCHES, "Branches"},
    {PerformanceCounters::DIVERGENT_BRANCHES, "Divergent Branches"},
    {PerformanceCounters::BRANCH_RECONVERGENCE_POINTS, "Branch Reconvergence Points"},
    
    // Cache statistics
    {PerformanceCounters::INSTRUCTION_CACHE_HITS, "Instruction Cache Hits"},
    {PerformanceCounters::INSTRUCTION_CACHE_MISSES, "Instruction Cache Misses"},
    {PerformanceCounters::DATA_CACHE_HITS, "Data Cache Hits"},
    {PerformanceCounters::DATA_CACHE_MISSES, "Data Cache Misses"}
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
        for (size_t i = 0; i < PerformanceCounters::MAX_COUNTER_ID; ++i) {
            m_counterValues[i] = 0;
        }
    }
    
    // Increment a counter by a specific value
    void increment(CounterID counterID, CounterValue value) {
        if (counterID < PerformanceCounters::MAX_COUNTER_ID) {
            m_counterValues[counterID] += value;
        }
    }
    
    // Get the value of a counter
    CounterValue getCounterValue(CounterID counterID) const {
        if (counterID < PerformanceCounters::MAX_COUNTER_ID) {
            return m_counterValues[counterID];
        }
        return 0;
    }
    
private:
    // Counter values
    CounterValue m_counterValues[PerformanceCounters::MAX_COUNTER_ID];
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
        if (counterID != INVALID) {  // Skip invalid counter
            const std::string& name = entry.second;
            CounterValue value = getCounterValue(counterID);
            
            // Skip counters with zero value
            if (value > 0) {
                std::cout << std::left << std::setw(30) << name << ": " << value << std::endl;
            }
        }
    }
    
    // Calculate and print IPC if cycles > 0
    CounterValue instructions = getCounterValue(INSTRUCTIONS_EXECUTED);
    CounterValue cycles = getCounterValue(CYCLES);
    if (cycles > 0) {
        double ipc = static_cast<double>(instructions) / cycles;
        std::cout << std::left << std::setw(30) << "Instructions Per Cycle (IPC)" << ": " << ipc << std::endl;
    }
}

// Initialize divergence statistics
void PerformanceCounters::initDivergenceStats() {
    divergenceStats.numDivergentPaths = 0;
    divergenceStats.maxDivergenceDepth = 0;
    divergenceStats.averageDivergenceRate = 0.0;
    divergenceStats.averageReconvergenceTime = 0.0;
    divergenceStats.divergenceImpactFactor = 0.0;
}

// Reset divergence statistics
void PerformanceCounters::resetDivergenceStats() {
    initDivergenceStats();
}

// Initialize branch statistics
void PerformanceCounters::initBranchStats() {
    branchStats.totalBranches = 0;
    branchStats.unconditionalBranches = 0;
    branchStats.divergentBranches = 0;
    branchStats.errors = 0;
}

// Reset branch statistics
void PerformanceCounters::resetBranchStats() {
    initBranchStats();
}
