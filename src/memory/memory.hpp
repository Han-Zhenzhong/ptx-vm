#ifndef MEMORY_HPP
#define MEMORY_HPP

#include <vector>
#include <unordered_map>
#include <cstdint>
#include <memory>

// Define memory space types
typedef uint64_t AddressSpace;

// Forward declarations
class MemorySubsystem;

// Memory access flags
enum class MemoryAccessFlags {
    READ,
    WRITE,
    EXECUTE
};

// Page fault handler function pointer type
typedef void (*PageFaultHandler)(uint64_t virtualAddress);

enum class MemorySpace {
    GENERIC,    // Generic address space
    GLOBAL,     // Global memory
    SHARED,     // Shared memory
    LOCAL,      // Local memory
    PARAMETER,  // Parameter memory
};

// TLB entry structure
struct TlbEntry {
    uint64_t virtualPage;
    uint64_t physicalPage;
    bool valid;
    bool dirty;
    uint64_t lastAccessed;
};

// Page table entry structure
struct PageTableEntry {
    uint64_t physicalPage;
    bool present;
    bool writable;
    bool dirty;
    uint64_t accessed;
};

// TLB configuration
struct TLBConfig {
    size_t size;
    bool enabled;
    uint64_t pageSize;
};

// Memory access result structure
struct MemoryAccessResult {
    bool success;
    bool pageFault;
    uint64_t physicalAddress;
    uint64_t virtualAddress;
    bool tlbHit;
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
    const void* getMemoryBuffer(MemorySpace space) const;

    // Get size of a memory space
    size_t getMemorySize(MemorySpace space) const;

    // TLB operations
    void configureTlb(const TLBConfig& config);
    bool translateAddress(uint64_t virtualAddress, uint64_t& physicalAddress);
    void flushTlb();
    
    // Page fault handling
    void setPageFaultHandler(const PageFaultHandler& handler);
    void handlePageFault(uint64_t virtualAddress);
    
    // Memory access with virtual memory support
    MemoryAccessResult accessMemory(uint64_t virtualAddress, MemoryAccessFlags flags);
    
    // Page table management
    void mapPage(uint64_t virtualPage, uint64_t physicalPage);
    void unmapPage(uint64_t virtualPage);
    
    // Cache operations
    void configureCache(const CacheConfig& config);
    void configureSharedMemory(const SharedMemoryConfig& config);
    
    // Shared memory operations
    size_t getBankConflicts(const std::vector<uint64_t>& addresses);
    
    // Performance statistics
    size_t getTlbHits() const;
    size_t getTlbMisses() const;
    size_t getPageFaults() const;
    size_t getCacheHits() const;
    size_t getCacheMisses() const;
    size_t getBankConflictsCount() const;
    
    // Destructor
    ~MemorySubsystem();
    
private:
    // Private implementation details
    class Impl;
    std::unique_ptr<Impl> pImpl;
};

#endif // MEMORY_HPP