#include "memory.hpp"
#include <stdexcept>
#include <cstring>
#include <algorithm>
#include <iostream>
#include "memory_optimizer.hpp"
#include <cstddef>  // For size_t
#include <cstdint>  // For uintX_t types

// Private implementation class
class MemorySubsystem::Impl {
public:
    // Structure to represent a memory space
    struct MemorySpaceInfo {
        void* buffer;         // Pointer to memory buffer
        size_t size;          // Size of the memory space
        bool ownsBuffer;      // Does this class own the buffer?
    };
    
    // Template methods for memory access
    template <typename T>
    T read(MemorySpace space, AddressSpace address) const {
        auto it = memorySpaces.find(space);
        if (it == memorySpaces.end()) {
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
    void write(MemorySpace space, AddressSpace address, const T& value) {
        auto it = memorySpaces.find(space);
        if (it == memorySpaces.end()) {
            throw std::invalid_argument("Invalid memory space");
        }
    
        if (address + sizeof(T) > it->second.size) {
            throw std::out_of_range("Memory address out of range");
        }
    
        std::memcpy(static_cast<uint8_t*>(it->second.buffer) + address, &value, sizeof(T));
    }

    // Memory space information
    std::unordered_map<MemorySpace, MemorySpaceInfo> memorySpaces;
    
    // TLB and virtual memory support
    std::vector<TlbEntry> tlb;
    std::unordered_map<uint64_t, PageTableEntry> pageTable;
    TLBConfig tlbConfig;
    PageFaultHandler pageFaultHandler;
    
    // Page size (4KB by default)
    static const uint64_t PAGE_SIZE = 4096;
    
    // Cache configuration
    CacheConfig cacheConfig;
    
    // Shared memory configuration
    SharedMemoryConfig sharedMemoryConfig;
    
    // Performance counters (moved from header)
    mutable size_t m_tlbHits_impl = 0;
    mutable size_t m_tlbMisses_impl = 0;
    mutable size_t m_pageFaults_impl = 0;
    mutable size_t m_cacheHits_impl = 0;
    mutable size_t m_cacheMisses_impl = 0;
    mutable size_t m_bankConflicts_impl = 0;
    
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

        Impl::MemorySpaceInfo info;
        info.buffer = globalBuffer;
        info.size = globalMemorySize;
        info.ownsBuffer = true;
        pImpl->memorySpaces[MemorySpace::GLOBAL] = info;
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

        Impl::MemorySpaceInfo info;
        info.buffer = sharedBuffer;
        info.size = sharedMemorySize;
        info.ownsBuffer = true;
        pImpl->memorySpaces[MemorySpace::SHARED] = info;
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

        Impl::MemorySpaceInfo info;
        info.buffer = localBuffer;
        info.size = localMemorySize;
        info.ownsBuffer = true;
        pImpl->memorySpaces[MemorySpace::LOCAL] = info;
    }
    
    // Initialize parameter memory (for kernel parameters)
    // Allocate 4KB for parameter memory (sufficient for most kernels)
    const size_t parameterMemorySize = 4 * 1024;
    void* parameterBuffer = new uint8_t[parameterMemorySize];
    if (!parameterBuffer) {
        // Clean up previously allocated memory
        if (globalMemorySize > 0) {
            delete[] static_cast<uint8_t*>(pImpl->memorySpaces[MemorySpace::GLOBAL].buffer);
        }
        if (sharedMemorySize > 0) {
            delete[] static_cast<uint8_t*>(pImpl->memorySpaces[MemorySpace::SHARED].buffer);
        }
        if (localMemorySize > 0) {
            delete[] static_cast<uint8_t*>(pImpl->memorySpaces[MemorySpace::LOCAL].buffer);
        }
        return false; // Allocation failed
    }
    
    Impl::MemorySpaceInfo paramInfo;
    paramInfo.buffer = parameterBuffer;
    paramInfo.size = parameterMemorySize;
    paramInfo.ownsBuffer = true;
    pImpl->memorySpaces[MemorySpace::PARAMETER] = paramInfo;

    return true;
}

#ifndef MEMORY_TEMPLATE_HPP
#define MEMORY_TEMPLATE_HPP

#include "memory.hpp"

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

#endif // MEMORY_TEMPLATE_HPP

// Explicit template instantiations for commonly used types
template uint8_t MemorySubsystem::read<uint8_t>(MemorySpace, AddressSpace) const;
template uint16_t MemorySubsystem::read<uint16_t>(MemorySpace, AddressSpace) const;
template uint32_t MemorySubsystem::read<uint32_t>(MemorySpace, AddressSpace) const;
template uint64_t MemorySubsystem::read<uint64_t>(MemorySpace, AddressSpace) const;
template int32_t MemorySubsystem::read<int32_t>(MemorySpace, AddressSpace) const;
template void MemorySubsystem::write<uint8_t>(MemorySpace, AddressSpace, const uint8_t&);
template void MemorySubsystem::write<uint16_t>(MemorySpace, AddressSpace, const uint16_t&);
template void MemorySubsystem::write<uint32_t>(MemorySpace, AddressSpace, const uint32_t&);
template void MemorySubsystem::write<uint64_t>(MemorySpace, AddressSpace, const uint64_t&);
template void MemorySubsystem::write<int32_t>(MemorySpace, AddressSpace, const int32_t&);

void* MemorySubsystem::getMemoryBuffer(MemorySpace space) {
    auto it = pImpl->memorySpaces.find(space);
    if (it == pImpl->memorySpaces.end()) {
        return nullptr;
    }

    return it->second.buffer;
}

const void* MemorySubsystem::getMemoryBuffer(MemorySpace space) const {
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
void MemorySubsystem::configureTlb(const TLBConfig& config) {
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
        pImpl->m_tlbHits_impl++;
        physicalAddress = (physicalPage * pImpl->tlbConfig.pageSize) + pageOffset;
        return true;
    }
    
    // TLB miss
    pImpl->m_tlbMisses_impl++;
    
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
    pImpl->m_pageFaults_impl++;
    
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
    return pImpl->m_tlbHits_impl;
}

size_t MemorySubsystem::getTlbMisses() const {
    return pImpl->m_tlbMisses_impl;
}

// Page fault handling
void MemorySubsystem::setPageFaultHandler(const PageFaultHandler& handler) {
    pImpl->pageFaultHandler = handler;
}

void MemorySubsystem::handlePageFault(uint64_t virtualAddress) {
    pImpl->m_pageFaults_impl++;
    
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
        result.tlbHit = (pImpl->m_tlbMisses_impl > 0); // Simplified check
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

// Cache operations
void MemorySubsystem::configureCache(const CacheConfig& config) {
    pImpl->cacheConfig = config;
}

void MemorySubsystem::configureSharedMemory(const SharedMemoryConfig& config) {
    pImpl->sharedMemoryConfig = config;
}

// Shared memory operations
size_t MemorySubsystem::getBankConflicts(const std::vector<uint64_t>& addresses) {
    if (pImpl->sharedMemoryConfig.bankCount == 0) {
        return 0;
    }
    
    // Count bank conflicts
    std::unordered_map<size_t, size_t> bankAccess;
    size_t conflicts = 0;
    
    for (uint64_t addr : addresses) {
        size_t bank = (addr / pImpl->sharedMemoryConfig.bankWidth) % pImpl->sharedMemoryConfig.bankCount;
        bankAccess[bank]++;
    }
    
    // Count conflicts (accesses beyond the first to each bank)
    for (const auto& entry : bankAccess) {
        if (entry.second > 1) {
            conflicts += (entry.second - 1);
        }
    }
    
    pImpl->m_bankConflicts_impl += conflicts;
    return conflicts;
}

// Performance statistics getters
size_t MemorySubsystem::getPageFaults() const {
    return pImpl->m_pageFaults_impl;
}

size_t MemorySubsystem::getCacheHits() const {
    return pImpl->m_cacheHits_impl;
}

size_t MemorySubsystem::getCacheMisses() const {
    return pImpl->m_cacheMisses_impl;
}

size_t MemorySubsystem::getBankConflictsCount() const {
    return pImpl->m_bankConflicts_impl;
}