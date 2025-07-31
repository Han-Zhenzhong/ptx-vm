#include "vm.hpp"
#include "performance_counters.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>

// Start profiling session
bool PTXVM::startProfiling(const std::string& profileName) {
    // Reset performance counters
    m_performanceCounters->reset();
    
    // Create a new profiler output file
    m_profileOutputFile = profileName;
    m_profileOutputStream.open(m_profileOutputFile, std::ios::out | std::ios::trunc);
    
    if (!m_profileOutputStream.is_open()) {
        std::cerr << "Failed to open profile output file: " << profileName << std::endl;
        return false;
    }
    
    // Write header information
    m_profileOutputStream << "PTX Virtual Machine Profiling Results" << std::endl;
    m_profileOutputStream << "Profile started at: " << getCurrentTime() << std::endl;
    m_profileOutputStream << "----------------------------------------" << std::endl;
    m_profileOutputStream << std::endl;
    
    return true;
}

// Stop profiling session
bool PTXVM::stopProfiling() {
    if (!m_profileOutputStream.is_open()) {
        return false;
    }
    
    // Write profiling results
    m_profileOutputStream << "Performance Counters:" << std::endl;
    m_profileOutputStream << "----------------------" << std::endl;
    
    // Get the performance counters
    const PerformanceCounters& counters = *m_performanceCounters;
    
    // Print all counters in CSV format
    m_profileOutputStream << "Counter Name,Value" << std::endl;
    
    // Helper macro for printing counter values
#define PRINT_COUNTER(name) \
    m_profileOutputStream << #name "," << counters.getCount(PerformanceCounters::name) << std::endl;

    // Output all counters
    PRINT_COUNTER(CYCLES)
    PRINT_COUNTER(INSTRUCTIONS_EXECUTED)
    PRINT_COUNTER(INSTRUCTION_FETCHES)
    PRINT_COUNTER(REGISTER_READS)
    PRINT_COUNTER(REGISTER_WRITES)
    PRINT_COUNTER(GLOBAL_MEMORY_READS)
    PRINT_COUNTER(GLOBAL_MEMORY_WRITES)
    PRINT_COUNTER(SHARED_MEMORY_READS)
    PRINT_COUNTER(SHARED_MEMORY_WRITES)
    PRINT_COUNTER(LOCAL_MEMORY_READS)
    PRINT_COUNTER(LOCAL_MEMORY_WRITES)
    PRINT_COUNTER(BRANCHES)
    PRINT_COUNTER(DIVERGENT_BRANCHES)
    PRINT_COUNTER(WARP_SWITCHES)
    PRINT_COUNTER(THREAD_SYNCHRONIZATION_EVENTS)
    PRINT_COUNTER(ADD_INSTRUCTIONS)
    PRINT_COUNTER(SUB_INSTRUCTIONS)
    PRINT_COUNTER(MUL_INSTRUCTIONS)
    PRINT_COUNTER(DIV_INSTRUCTIONS)
    PRINT_COUNTER(MOV_INSTRUCTIONS)
    PRINT_COUNTER(LD_INSTRUCTIONS)
    PRINT_COUNTER(ST_INSTRUCTIONS)
    PRINT_COUNTER(BRA_INSTRUCTIONS)
    PRINT_COUNTER(EXIT_INSTRUCTIONS)
    PRINT_COUNTER(NOP_INSTRUCTIONS)
    PRINT_COUNTER(SPILL_OPERATIONS)
    PRINT_COUNTER(FETCH_STALLS)
    PRINT_COUNTER(EXECUTION_STALLS)
    PRINT_COUNTER(CONTEXT_SWITCHES)
    PRINT_COUNTER(ICACHE_MISSES)
    PRINT_COUNTER(DCACHE_MISSES)
    PRINT_COUNTER(SCACHE_MISSES)
    PRINT_COUNTER(TLB_MISSES)
    PRINT_COUNTER(PAGE_FAULTS)
    PRINT_COUNTER(MEMORY_BW_USED)
    PRINT_COUNTER(COMPUTE_UNITS_UTILIZED)
    PRINT_COUNTER(REGISTERS_UTILIZED)
    PRINT_COUNTER(INSTRUCTION_QUEUE_DEPTH)
    PRINT_COUNTER(MEMORY_QUEUE_DEPTH)
    PRINT_COUNTER(WARP_INST_LATENCY)
    PRINT_COUNTER(STALL_CYCLES)
    PRINT_COUNTER(OCCUPANCY)
    PRINT_COUNTER(UTILIZATION)
    PRINT_COUNTER(INSTRUCTION_LATENCY)
    PRINT_COUNTER(MEMORY_LATENCY)
    PRINT_COUNTER(CACHE_HIT_RATE)
    PRINT_COUNTER(MEMORY_BANDWIDTH)
    PRINT_COUNTER(POWER_CONSUMPTION)
    PRINT_COUNTER(TEMPERATURE)
#undef PRINT_COUNTER
    
    // Calculate and print derived metrics
    double ipc = static_cast<double>(counters.getCount(PerformanceCounters::INSTRUCTIONS_EXECUTED)) / 
                 counters.getCount(PerformanceCounters::CYCLES);
    
    m_profileOutputStream << std::endl;
    m_profileOutputStream << "Derived Metrics:" << std::endl;
    m_profileOutputStream << "------------------" << std::endl;
    m_profileOutputStream << "IPC (Instructions Per Cycle)," << ipc << std::endl;
    m_profileOutputStream << "Execution Time (cycles)," << counters.getCount(PerformanceCounters::CYCLES) << std::endl;
    
    // Add more derived metrics as needed
    
    // Close the output stream
    m_profileOutputStream.close();
    return true;
}

// Get current time as string
std::string PTXVM::getCurrentTime() {
    time_t now = time(nullptr);
    char buffer[80];
    strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", localtime(&now));
    return std::string(buffer);
}

// Dump execution statistics to console
void PTXVM::dumpExecutionStats() {
    const PerformanceCounters& counters = getPerformanceCounters();
    
    std::cout << "Execution Statistics:" << std::endl;
    std::cout << "-------------------" << std::endl;
    std::cout << "Total Cycles:            " << counters.getCount(PerformanceCounters::CYCLES) << std::endl;
    std::cout << "Instructions Executed:     " << counters.getCount(PerformanceCounters::INSTRUCTIONS_EXECUTED) << std::endl;
    
    double ipc = static_cast<double>(counters.getCount(PerformanceCounters::INSTRUCTIONS_EXECUTED)) / 
                 counters.getCount(PerformanceCounters::CYCLES);
    
    std::cout << "IPC (Instructions/Cycle): " << ipc << std::endl;
    
    // Register allocation stats
    std::cout << "Register Utilization:     " << m_registerAllocator->getRegisterUtilization() * 100 << "%" << std::endl;
    std::cout << "Spill Operations:        " << counters.getCount(PerformanceCounters::SPILL_OPERATIONS) << std::endl;
    
    // Memory access stats
    std::cout << "Global Memory Reads:      " << counters.getCount(PerformanceCounters::GLOBAL_MEMORY_READS) << std::endl;
    std::cout << "Global Memory Writes:     " << counters.getCount(PerformanceCounters::GLOBAL_MEMORY_WRITES) << std::endl;
    std::cout << "Shared Memory Reads:      " << counters.getCount(PerformanceCounters::SHARED_MEMORY_READS) << std::endl;
    std::cout << "Shared Memory Writes:     " << counters.getCount(PerformanceCounters::SHARED_MEMORY_WRITES) << std::endl;
    std::cout << "Local Memory Reads:       " << counters.getCount(PerformanceCounters::LOCAL_MEMORY_READS) << std::endl;
    std::cout << "Local Memory Writes:      " << counters.getCount(PerformanceCounters::LOCAL_MEMORY_WRITES) << std::endl;
    
    // Control flow stats
    std::cout << "Branches:                " << counters.getCount(PerformanceCounters::BRANCHES) << std::endl;
    std::cout << "Divergent Branches:      " << counters.getCount(PerformanceCounters::DIVERGENT_BRANCHES) << std::endl;
    std::cout << "Warp Switches:           " << counters.getCount(PerformanceCounters::WARP_SWITCHES) << std::endl;
    
    // Additional metrics
    std::cout << "TLB Misses:              " << counters.getCount(PerformanceCounters::TLB_MISSES) << std::endl;
    std::cout << "Page Faults:             " << counters.getCount(PerformanceCounters::PAGE_FAULTS) << std::endl;
    std::cout << "Cache Hit Rate:          " << counters.getCount(PerformanceCounters::CACHE_HIT_RATE) << std::endl;
    
    std::cout << std::endl;
}

// Dump instruction mix analysis
void PTXVM::dumpInstructionMixAnalysis() {
    const PerformanceCounters& counters = getPerformanceCounters();
    size_t totalInstructions = counters.getCount(PerformanceCounters::INSTRUCTIONS_EXECUTED);
    
    if (totalInstructions == 0) {
        std::cout << "No instructions executed." << std::endl;
        return;
    }
    
    std::cout << "Instruction Mix Analysis:" << std::endl;
    std::cout << "--------------------------" << std::endl;
    
    // Calculate percentages for each instruction type
    auto printPercentage = [this, &counters, totalInstructions](const char* name, PerformanceCounters::CounterType type) {
        size_t count = counters.getCount(type);
        double percentage = (static_cast<double>(count) / totalInstructions) * 100.0;
        m_profileOutputStream << name << "," << count << "," << percentage << std::endl;
    };
    
    // Open file for writing detailed analysis if we're profiling
    std::ofstream detailStream;
    if (m_profileOutputStream.is_open()) {
        detailStream.open("profile/instruction_mix_analysis.csv");
        if (detailStream.is_open()) {
            detailStream << "Instruction Type,Count,Percentage (%)" << std::endl;
            
            printPercentage("ADD", PerformanceCounters::ADD_INSTRUCTIONS);
            printPercentage("SUB", PerformanceCounters::SUB_INSTRUCTIONS);
            printPercentage("MUL", PerformanceCounters::MUL_INSTRUCTIONS);
            printPercentage("DIV", PerformanceCounters::DIV_INSTRUCTIONS);
            printPercentage("MOV", PerformanceCounters::MOV_INSTRUCTIONS);
            printPercentage("LD", PerformanceCounters::LD_INSTRUCTIONS);
            printPercentage("ST", PerformanceCounters::ST_INSTRUCTIONS);
            printPercentage("BRA", PerformanceCounters::BRA_INSTRUCTIONS);
            printPercentage("EXIT", PerformanceCounters::EXIT_INSTRUCTIONS);
            printPercentage("NOP", PerformanceCounters::NOP_INSTRUCTIONS);
            
            detailStream.close();
        }
    }
    
    // Also print summary to console
    std::cout << "Instruction Mix Summary:" << std::endl;
    std::cout << "-------------------------" << std::endl;
    
    auto printToConsole = [](const char* name, const PerformanceCounters& counters, size_t total) {
        size_t count = counters.getCount(PerformanceCounters::ADD_INSTRUCTIONS);
        double percentage = (static_cast<double>(count) / total) * 100.0;
        std::cout << name << ": " << count << " (" << percentage << "%)" << std::endl;
    };
    
    printToConsole("ADD", counters, totalInstructions);
    printToConsole("SUB", counters, totalInstructions);
    printToConsole("MUL", counters, totalInstructions);
    printToConsole("DIV", counters, totalInstructions);
    printToConsole("MOV", counters, totalInstructions);
    printToConsole("LD", counters, totalInstructions);
    printToConsole("ST", counters, totalInstructions);
    printToConsole("BRA", counters, totalInstructions);
    printToConsole("EXIT", counters, totalInstructions);
    printToConsole("NOP", counters, totalInstructions);
    
    std::cout << std::endl;
}

// Dump memory access pattern analysis
void PTXVM::dumpMemoryAccessAnalysis() {
    const PerformanceCounters& counters = getPerformanceCounters();
    
    std::cout << "Memory Access Analysis:" << std::endl;
    std::cout << "-----------------------" << std::endl;
    
    // Total memory operations
    size_t totalMemOps = counters.getCount(PerformanceCounters::GLOBAL_MEMORY_READS) + 
                        counters.getCount(PerformanceCounters::GLOBAL_MEMORY_WRITES) + 
                        counters.getCount(PerformanceCounters::SHARED_MEMORY_READS) + 
                        counters.getCount(PerformanceCounters::SHARED_MEMORY_WRITES) + 
                        counters.getCount(PerformanceCounters::LOCAL_MEMORY_READS) + 
                        counters.getCount(PerformanceCounters::LOCAL_MEMORY_WRITES);
    
    // Global memory
    size_t globalReads = counters.getCount(PerformanceCounters::GLOBAL_MEMORY_READS);
    size_t globalWrites = counters.getCount(PerformanceCounters::GLOBAL_MEMORY_WRITES);
    
    std::cout << "Global Memory: " << globalReads + globalWrites << " accesses" << std::endl;
    std::cout << "  Reads:   " << globalReads << " (" 
              << (globalReads * 100.0) / totalMemOps << "%)" << std::endl;
    std::cout << "  Writes:  " << globalWrites << " (" 
              << (globalWrites * 100.0) / totalMemOps << "%)" << std::endl;
    
    // Shared memory
    size_t sharedReads = counters.getCount(PerformanceCounters::SHARED_MEMORY_READS);
    size_t sharedWrites = counters.getCount(PerformanceCounters::SHARED_MEMORY_WRITES);
    
    std::cout << "Shared Memory: " << sharedReads + sharedWrites << " accesses" << std::endl;
    std::cout << "  Reads:   " << sharedReads << " (" 
              << (sharedReads * 100.0) / totalMemOps << "%)" << std::endl;
    std::cout << "  Writes:  " << sharedWrites << " (" 
              << (sharedWrites * 100.0) / totalMemOps << "%)" << std::endl;
    
    // Local memory
    size_t localReads = counters.getCount(PerformanceCounters::LOCAL_MEMORY_READS);
    size_t localWrites = counters.getCount(PerformanceCounters::LOCAL_MEMORY_WRITES);
    
    std::cout << "Local Memory: " << localReads + localWrites << " accesses" << std::endl;
    std::cout << "  Reads:   " << localReads << " (" 
              << (localReads * 100.0) / totalMemOps << "%)" << std::endl;
    std::cout << "  Writes:  " << localWrites << " (" 
              << (localWrites * 100.0) / totalMemOps << "%)" << std::endl;
    
    // Cache analysis
    size_t icacheMisses = counters.getCount(PerformanceCounters::ICACHE_MISSES);
    size_t dcacheMisses = counters.getCount(PerformanceCounters::DCACHE_MISSES);
    size_t scacheMisses = counters.getCount(PerformanceCounters::SCACHE_MISSES);
    
    std::cout << std::endl;
    std::cout << "Cache Analysis:" << std::endl;
    std::cout << "----------------" << std::endl;
    
    std::cout << "Instruction Cache Misses: " << icacheMisses << std::endl;
    std::cout << "Data Cache Misses:       " << dcacheMisses << std::endl;
    std::cout << "Shared Cache Misses:     " << scacheMisses << std::endl;
    
    // Memory bandwidth
    size_t memoryBW = counters.getCount(PerformanceCounters::MEMORY_BANDWIDTH);
    std::cout << "Estimated Memory Bandwidth Used: " << memoryBW << " bytes/cycle" << std::endl;
    
    // TLB misses
    size_t tlbMisses = counters.getCount(PerformanceCounters::TLB_MISSES);
    std::cout << "TLB Misses: " << tlbMisses << std::endl;
    
    std::cout << std::endl;
}

// Dump warp execution analysis
void PTXVM::dumpWarpExecutionAnalysis() {
    const WarpScheduler& scheduler = getExecutor().getWarpScheduler();
    
    std::cout << "Warp Execution Analysis:" << std::endl;
    std::cout << "--------------------------" << std::endl;
    
    // Get warp count
    uint32_t numWarps = scheduler.getNumWarps();
    std::cout << "Total Warps: " << numWarps << std::endl;
    
    // For each warp, get some basic statistics
    size_t totalInstructions = 0;
    size_t activeWarps = 0;
    
    for (uint32_t warpId = 0; warpId < numWarps; ++warpId) {
        // Check if this warp has any work
        bool hasWork = false;
        for (size_t i = 0; i < 10 && !hasWork; ++i) {
            hasWork = (scheduler.getActiveThreads(warpId) != 0);
            if (!hasWork) {
                break;
            }
        }
        
        if (hasWork) {
            activeWarps++;
            
            // In real implementation, we would track per-warp instruction counts
            // This is just a placeholder
            size_t instructionsForWarp = 0;  // Would come from warp statistics
            totalInstructions += instructionsForWarp;
            
            // Print basic warp info
            std::cout << "Warp " << warpId << ": " << instructionsForWarp << " instructions" << std::endl;
        }
    }
    
    // Calculate average IPC per warp
    double totalCycles = getPerformanceCounters().getCount(PerformanceCounters::CYCLES);
    double totalInstructionsExecuted = getPerformanceCounters().getCount(PerformanceCounters::INSTRUCTIONS_EXECUTED);
    
    double overallIPC = (totalInstructionsExecuted > 0 && totalCycles > 0) ? 
                       (totalInstructionsExecuted / totalCycles) : 0;
    
    std::cout << "\nOverall IPC: " << overallIPC << std::endl;
    std::cout << "Active warps: " << activeWarps << " out of " << numWarps << std::endl;
    
    // Calculate warp divergence
    size_t divergentBranches = getPerformanceCounters().getCount(PerformanceCounters::DIVERGENT_BRANCHES);
    size_t totalBranches = getPerformanceCounters().getCount(PerformanceCounters::BRANCHES);
    
    if (totalBranches > 0) {
        double divergenceRate = (static_cast<double>(divergentBranches) / totalBranches) * 100.0;
        std::cout << "Divergence rate: " << divergenceRate << "%" << std::endl;
    }
    
    // Register usage
    double registerUtilization = m_registerAllocator->getRegisterUtilization() * 100.0;
    std::cout << "Register utilization: " << registerUtilization << "%" << std::endl;
    
    std::cout << std::endl;
}