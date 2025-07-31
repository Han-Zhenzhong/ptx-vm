#include "memory.hpp"
#include <stdexcept>
#include <cstring>
#include <algorithm>
#include <iostream>

// Private implementation class
class MemorySubsystem::Impl {
public:
    // Structure to represent a memory space
    struct MemorySpaceInfo {
        void* buffer;         // Pointer to memory buffer
        size_t size;          // Size of the memory space
        bool ownsBuffer;      // Does this class own the buffer?
    };

    // Memory space information
    std::unordered_map<MemorySpace, MemorySpaceInfo> memorySpaces;
    
    // TLB and virtual memory support
    std::vector<TlbEntry> tlb;
    std::unordered_map<uint64_t, PageTableEntry> pageTable;
    TlbConfig tlbConfig;
    PageFaultHandler pageFaultHandler;
    
    // TLB statistics
    size_t tlbHits = 0;
    size_t tlbMisses = 0;
    size_t pageFaults = 0;
    
    // Page size (4KB by default)
    static const uint64_t PAGE_SIZE = 4096;
    
    // Private methods for TLB and virtual memory
    bool lookupTlb(uint64_t virtualPage, uint64_t& physicalPage) {
        for (const auto& entry : tlb) {
            if (entry.valid && entry.virtualPage == virtualPage) {
                physicalPage = entry.physicalPage;
                return true;
            }
        }
        return false;
    }
    
    void updateTlb(uint64_t virtualPage, uint64_t physicalPage) {
        // Find an invalid entry or evict the oldest entry
        size_t oldestIndex = 0;
        uint64_t oldestTime = UINT64_MAX;
        
        for (size_t i = 0; i < tlb.size(); ++i) {
            if (!tlb[i].valid) {
                // Use this invalid entry
                tlb[i].valid = true;
                tlb[i].virtualPage = virtualPage;
                tlb[i].physicalPage = physicalPage;
                tlb[i].dirty = false;
                tlb[i].lastAccessed = 0; // TODO: Use real timestamp
                return;
            }
            
            if (tlb[i].lastAccessed < oldestTime) {
                oldestTime = tlb[i].lastAccessed;
                oldestIndex = i;
            }
        }
        
        // Evict the oldest entry
        tlb[oldestIndex].valid = true;
        tlb[oldestIndex].virtualPage = virtualPage;
        tlb[oldestIndex].physicalPage = physicalPage;
        tlb[oldestIndex].dirty = false;
        tlb[oldestIndex].lastAccessed = 0; // TODO: Use real timestamp
    }
    
    uint64_t getVirtualPage(uint64_t virtualAddress) const {
        return virtualAddress / PAGE_SIZE;
    }
    
    uint64_t getPageOffset(uint64_t virtualAddress) const {
        return virtualAddress % PAGE_SIZE;
    }
    
    MemoryAccessResult performPhysicalAccess(uint64_t physicalAddress, MemoryAccessFlags flags) {
        MemoryAccessResult result;
        result.success = true;
        result.pageFault = false;
        result.physicalAddress = physicalAddress;
        result.tlbHit = false; // This is set by the caller
        return result;
    }
};

MemorySubsystem::MemorySubsystem() : pImpl(std::make_unique<Impl>()) {
    // Initialize default TLB config
    pImpl->tlbConfig.size = 32; // 32-entry TLB
    pImpl->tlbConfig.enabled = true;
    pImpl->tlbConfig.pageSize = Impl::PAGE_SIZE;
    
    // Initialize TLB
    pImpl->tlb.resize(pImpl->tlbConfig.size);
    for (auto& entry : pImpl->tlb) {
        entry.valid = false;
        entry.virtualPage = 0;
        entry.physicalPage = 0;
        entry.dirty = false;
        entry.lastAccessed = 0;
    }
}

MemorySubsystem::~MemorySubsystem() = default;

bool MemorySubsystem::initialize(size_t globalMemorySize, 
                                 size_t sharedMemorySize,
                                 size_t localMemorySize)
{
    // Initialize global memory
    if (globalMemorySize > 0) {
        void* globalBuffer = new uint8_t[globalMemorySize];
        if (!globalBuffer) {
            return false; // Allocation failed
        }

        pImpl->memorySpaces[MemorySpace::GLOBAL] = {
            .buffer = globalBuffer,
            .size = globalMemorySize,
            .ownsBuffer = true
        };
    }

    // Initialize shared memory
    if (sharedMemorySize > 0) {
        void* sharedBuffer = new uint8_t[sharedMemorySize];
        if (!sharedBuffer) {
            // Clean up previously allocated memory
            if (globalMemorySize > 0) {
                delete[] static_cast<uint8_t*>(pImpl->memorySpaces[MemorySpace::GLOBAL].buffer);
            }
            return false; // Allocation failed
        }

        pImpl->memorySpaces[MemorySpace::SHARED] = {
            .buffer = sharedBuffer,
            .size = sharedMemorySize,
            .ownsBuffer = true
        };
    }

    // Initialize local memory
    if (localMemorySize > 0) {
        void* localBuffer = new uint8_t[localMemorySize];
        if (!localBuffer) {
            // Clean up previously allocated memory
            if (globalMemorySize > 0) {
                delete[] static_cast<uint8_t*>(pImpl->memorySpaces[MemorySpace::GLOBAL].buffer);
            }
            if (sharedMemorySize > 0) {
                delete[] static_cast<uint8_t*>(pImpl->memorySpaces[MemorySpace::SHARED].buffer);
            }
            return false; // Allocation failed
        }

        pImpl->memorySpaces[MemorySpace::LOCAL] = {
            .buffer = localBuffer,
            .size = localMemorySize,
            .ownsBuffer = true
        };
    }

    return true;
}

template <typename T>
T MemorySubsystem::read(MemorySpace space, AddressSpace address) const {
    auto it = pImpl->memorySpaces.find(space);
    if (it == pImpl->memorySpaces.end()) {
        throw std::invalid_argument("Invalid memory space");
    }

    if (address + sizeof(T) > it->second.size) {
        throw std::out_of_range("Memory address out of range");
    }

    T value;
    std::memcpy(&value, static_cast<uint8_t*>(it->second.buffer) + address, sizeof(T));
    return value;
}

template <typename T>
void MemorySubsystem::write(MemorySpace space, AddressSpace address, const T& value) {
    auto it = pImpl->memorySpaces.find(space);
    if (it == pImpl->memorySpaces.end()) {
        throw std::invalid_argument("Invalid memory space");
    }

    if (address + sizeof(T) > it->second.size) {
        throw std::out_of_range("Memory address out of range");
    }

    std::memcpy(static_cast<uint8_t*>(it->second.buffer) + address, &value, sizeof(T));
}

void* MemorySubsystem::getMemoryBuffer(MemorySpace space) {
    auto it = pImpl->memorySpaces.find(space);
    if (it == pImpl->memorySpaces.end()) {
        return nullptr;
    }

    return it->second.buffer;
}

size_t MemorySubsystem::getMemorySize(MemorySpace space) const {
    auto it = pImpl->memorySpaces.find(space);
    if (it == pImpl->memorySpaces.end()) {
        return 0;
    }

    return it->second.size;
}

// TLB management
void MemorySubsystem::configureTlb(const TlbConfig& config) {
    pImpl->tlbConfig = config;
    
    // Resize TLB if needed
    if (pImpl->tlb.size() != config.size) {
        pImpl->tlb.resize(config.size);
        
        // Initialize new entries
        for (auto& entry : pImpl->tlb) {
            entry.valid = false;
            entry.virtualPage = 0;
            entry.physicalPage = 0;
            entry.dirty = false;
            entry.lastAccessed = 0;
        }
    }
}

bool MemorySubsystem::translateAddress(uint64_t virtualAddress, uint64_t& physicalAddress) {
    if (!pImpl->tlbConfig.enabled) {
        // If TLB is disabled, use identity mapping
        physicalAddress = virtualAddress;
        return true;
    }
    
    uint64_t virtualPage = pImpl->getVirtualPage(virtualAddress);
    uint64_t pageOffset = pImpl->getPageOffset(virtualAddress);
    uint64_t physicalPage;
    
    // Try TLB lookup first
    if (pImpl->lookupTlb(virtualPage, physicalPage)) {
        // TLB hit
        pImpl->tlbHits++;
        physicalAddress = (physicalPage * pImpl->tlbConfig.pageSize) + pageOffset;
        return true;
    }
    
    // TLB miss
    pImpl->tlbMisses++;
    
    // Check page table
    auto it = pImpl->pageTable.find(virtualPage);
    if (it != pImpl->pageTable.end() && it->second.present) {
        // Page table hit
        physicalPage = it->second.physicalPage;
        
        // Update TLB
        pImpl->updateTlb(virtualPage, physicalPage);
        
        physicalAddress = (physicalPage * pImpl->tlbConfig.pageSize) + pageOffset;
        return true;
    }
    
    // Page fault
    pImpl->pageFaults++;
    
    // Call page fault handler if set
    if (pImpl->pageFaultHandler) {
        pImpl->pageFaultHandler(virtualAddress);
    }
    
    return false;
}

void MemorySubsystem::flushTlb() {
    for (auto& entry : pImpl->tlb) {
        entry.valid = false;
    }
}

size_t MemorySubsystem::getTlbHits() const {
    return pImpl->tlbHits;
}

size_t MemorySubsystem::getTlbMisses() const {
    return pImpl->tlbMisses;
}

// Page fault handling
void MemorySubsystem::setPageFaultHandler(const PageFaultHandler& handler) {
    pImpl->pageFaultHandler = handler;
}

void MemorySubsystem::handlePageFault(uint64_t virtualAddress) {
    pImpl->pageFaults++;
    
    // Call page fault handler if set
    if (pImpl->pageFaultHandler) {
        pImpl->pageFaultHandler(virtualAddress);
    }
}

// Memory access with virtual memory support
MemoryAccessResult MemorySubsystem::accessMemory(uint64_t virtualAddress, MemoryAccessFlags flags) {
    MemoryAccessResult result;
    result.virtualAddress = virtualAddress;
    result.pageFault = false;
    result.tlbHit = false;
    
    uint64_t physicalAddress;
    if (translateAddress(virtualAddress, physicalAddress)) {
        result.success = true;
        result.physicalAddress = physicalAddress;
        result.tlbHit = (pImpl->tlbMisses > 0); // Simplified check
        return pImpl->performPhysicalAccess(physicalAddress, flags);
    } else {
        result.success = false;
        result.pageFault = true;
        return result;
    }
}

// Page table management
void MemorySubsystem::mapPage(uint64_t virtualPage, uint64_t physicalPage) {
    PageTableEntry entry;
    entry.physicalPage = physicalPage;
    entry.present = true;
    entry.writable = true;
    entry.dirty = false;
    entry.accessed = 0;
    
    pImpl->pageTable[virtualPage] = entry;
}

void MemorySubsystem::unmapPage(uint64_t virtualPage) {
    pImpl->pageTable.erase(virtualPage);
}