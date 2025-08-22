#ifndef MEMORY_OPTIMIZER_HPP
#define MEMORY_OPTIMIZER_HPP

#include <cstdint>
#include <vector>
#include <memory>
#include "memory.hpp"

// Memory optimization types
typedef enum {
    CACHE_TYPE_INSTRUCTION = 0,  // Instruction cache
    CACHE_TYPE_DATA,             // Data cache
    CACHE_TYPE_SHARED,           // Shared memory cache
    CACHE_TYPE_LAST = CACHE_TYPE_SHARED
} CacheType;

// Use CacheConfig and TLBConfig from memory.hpp instead of redefining them

// Memory access pattern type
typedef enum {
    ACCESS_PATTERN_SEQUENTIAL = 0,
    ACCESS_PATTERN_STRIDED,
    ACCESS_PATTERN_RANDOM,
    ACCESS_PATTERN_LAST = ACCESS_PATTERN_RANDOM
} AccessPattern;

// Memory statistics structure
typedef struct {
    uint64_t icacheHits;        // Instruction cache hits
    uint64_t icacheMisses;      // Instruction cache misses
    uint64_t dcacheHits;        // Data cache hits
    uint64_t dcacheMisses;      // Data cache misses
    uint64_t scacheBankConflicts; // Shared memory bank conflicts
    uint64_t tlbHits;          // TLB hits
    uint64_t tlbMisses;        // TLB misses
    uint64_t pageFaults;       // Page faults
    uint64_t coalescedAccesses;   // Coalesced memory accesses
    uint64_t uncoalescedAccesses; // Uncoalesced memory accesses
} MemoryStats;

// Memory access information
typedef struct {
    MemorySpace space;        // Memory space being accessed
    uint64_t address;         // Address of the access
    size_t size;              // Size of the access
    uint64_t threadMask;      // Threads involved in access
    AccessPattern pattern;    // Type of access pattern
} MemoryAccessInfo;

// Memory optimization configuration
struct MemoryOptimizerConfig {
    // Cache configuration
    bool enableCaching;
    size_t cacheSize;
    size_t cacheLineSize;
    size_t cacheAssociativity;
    bool writeThrough;
    
    // Shared memory configuration
    bool enableSharedMemoryOptimization;
    size_t sharedMemoryBanks;
    size_t sharedMemoryBankWidth;
    
    // TLB configuration
    bool enableTLB;
    size_t tlbEntries;
    size_t pageSize;
    
    // Optimization level (0 = none, 1 = basic, 2 = advanced)
    int optimizationLevel;
};

class MemoryOptimizerFramework {
public:
    MemoryOptimizerFramework();
    ~MemoryOptimizerFramework();
    
    // Configure the memory optimizer
    void configure(const MemoryOptimizerConfig& config);
    
    // Get current configuration
    const MemoryOptimizerConfig& getConfig() const;
    
    // Analyze memory access patterns
    void analyzeAccessPattern(const std::vector<uint64_t>& addresses);
    
    // Get memory coalescing efficiency
    double getCoalescingEfficiency() const;
    
    // Get bank conflict count
    size_t getBankConflicts() const;
    
    // Get cache hit rate
    double getCacheHitRate() const;
    
    // Get TLB hit rate
    double getTlbHitRate() const;
    
    // Optimize memory access
    std::vector<uint64_t> optimizeAccessPattern(const std::vector<uint64_t>& addresses);
    
private:
    MemoryOptimizerConfig m_config;
    MemorySubsystem* m_memorySubsystem;
    
    // Statistics
    size_t m_totalAccesses;
    size_t m_coalescedAccesses;
    size_t m_bankConflicts;
    size_t m_cacheHits;
    size_t m_cacheMisses;
    size_t m_tlbHits;
    size_t m_tlbMisses;
};

class MemoryOptimizer {
public:
    // Constructor/destructor
    MemoryOptimizer();
    ~MemoryOptimizer();

    // Initialize with specified configurations
    bool initialize(const CacheConfig& dataCacheConfig,
                  const SharedMemoryConfig& sharedCacheConfig,
                  const TLBConfig& tlbConfig);

    // Reset to initial state
    void reset();

    // Check if address is cached
    bool isCached(CacheType cacheType, uint64_t address) const;

    // Check for memory coalescing opportunities
    bool checkCoalescing(uint64_t address, size_t size, uint64_t threadMask);

    // Check for shared memory bank conflicts
    bool checkBankConflict(uint64_t address, size_t size, uint64_t threadMask);

    // Translate virtual to physical address
    uint64_t translateAddress(uint64_t virtualAddress);

    // Handle page fault
    bool handlePageFault(uint64_t virtualAddress);

    // Get memory statistics
    const MemoryStats& getMemoryStats() const;

    // Print memory statistics
    void printStats() const;

private:
    // Private implementation details
    class Impl;
    std::unique_ptr<Impl> pImpl;
};

#endif // MEMORY_OPTIMIZER_HPP