#ifndef MEMORY_HPP
#define MEMORY_HPP

#include <vector>
#include <unordered_map>
#include <cstdint>
#include <memory>
#include "../include/logger.hpp"

// Define memory space types
typedef uint64_t AddressSpace;

enum class MemorySpace {
    GENERIC,    // Generic address space
    GLOBAL,     // Global memory
    SHARED,     // Shared memory
    LOCAL,      // Local memory
    PARAMETER,  // Parameter memory
};

// TLB entry structure
struct TlbEntry {
    uint64_t virtualPageNumber;
    uint64_t physicalPageNumber;
    bool valid;
    bool dirty;
    uint64_t lastAccessed;
};

// Page table entry structure
struct PageTableEntry {
    uint64_t physicalPageNumber;
    bool present;
    bool writable;
    bool dirty;
    uint64_t accessed;
};

// Memory access result structure
struct MemoryAccessResult {
    bool success;
    bool pageFault;
    uint64_t physicalAddress;
    uint64_t virtualAddress;
};

// Cache configuration structure
struct CacheConfig {
    size_t cacheSize;           // Total cache size in bytes
    size_t lineSize;            // Cache line size in bytes
    size_t associativity;       // Cache associativity (1 = direct mapped, 0 = fully associative)
    bool writeThrough;          // Write-through or write-back policy
    size_t replacementPolicy;   // 0 = LRU, 1 = FIFO, 2 = Random
};

// Shared memory bank configuration
struct SharedMemoryConfig {
    size_t bankCount;           // Number of banks
    size_t bankWidth;           // Width of each bank in bytes
    size_t bankSize;            // Size of each bank in bytes
};

class MemorySubsystem {
public:
    // Constructor
    MemorySubsystem();
    
    // Initialize memory subsystem
    bool initialize();
    
    // Initialize the memory subsystem
    bool initialize(size_t globalMemorySize = 1024 * 1024, 
                   size_t sharedMemorySize = 64 * 1024,
                   size_t localMemorySize = 64 * 1024);

    // Read from memory
    template <typename T>
    T read(MemorySpace space, AddressSpace address) const;

    // Write to memory
    template <typename T>
    void write(MemorySpace space, AddressSpace address, const T& value);

    // Get pointer to memory buffer for a specific space
    void* getMemoryBuffer(MemorySpace space);

    // Get size of a memory space
    size_t getMemorySize(MemorySpace space) const;

    // TLB operations
    void initializeTLB(size_t tlbSize);
    MemoryAccessResult translateAddress(uint64_t virtualAddress);
    bool handlePageFault(uint64_t virtualAddress);
    
    // Cache operations
    void configureCache(const CacheConfig& config);
    void configureSharedMemory(const SharedMemoryConfig& config);
    
    // Memory coalescing operations
    size_t calculateCoalescingEfficiency(const std::vector<uint64_t>& addresses);
    bool isAccessCoalesced(const std::vector<uint64_t>& addresses);
    
    // Shared memory operations
    size_t getBankConflicts(const std::vector<uint64_t>& addresses);
    
    // Performance statistics
    size_t getTlbHits() const { return m_tlbHits; }
    size_t getTlbMisses() const { return m_tlbMisses; }
    size_t getPageFaults() const { return m_pageFaults; }
    size_t getCacheHits() const { return m_cacheHits; }
    size_t getCacheMisses() const { return m_cacheMisses; }
    size_t getBankConflictsCount() const { return m_bankConflicts; }
    
private:
    // Private implementation details
    class Impl;
    std::unique_ptr<Impl> pImpl;

    // TLB and page fault handling
    std::vector<TlbEntry> m_tlb;
    std::unordered_map<uint64_t, PageTableEntry> m_pageTable;
    size_t m_tlbSize;
    size_t m_pageSize;
    size_t m_tlbHits;
    size_t m_tlbMisses;
    size_t m_pageFaults;
    
    // Cache configuration
    CacheConfig m_cacheConfig;
    SharedMemoryConfig m_sharedMemConfig;
    
    // Performance counters
    size_t m_cacheHits;
    size_t m_cacheMisses;
    size_t m_bankConflicts;
    
    // Private helper methods
    uint64_t getVirtualPageNumber(uint64_t virtualAddress) const;
    uint64_t getPageOffset(uint64_t virtualAddress) const;
    TlbEntry* findTlbEntry(uint64_t virtualPageNumber);
    void updateTlbEntry(TlbEntry* entry, uint64_t virtualPageNumber, uint64_t physicalPageNumber);
    uint64_t allocatePhysicalPage();
    void evictTlbEntry();
};

#endif // MEMORY_HPP