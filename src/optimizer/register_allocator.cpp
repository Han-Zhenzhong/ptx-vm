#include "register_allocator.hpp"
#include <iostream>
#include <stdexcept>
#include <algorithm>
#include "vm.hpp"

// Define invalid register constant
static const uint32_t INVALID_REGISTER = static_cast<uint32_t>(-1);

// Private implementation class
class RegisterAllocator::Impl {
public:
    Impl(PTXVM* vm) : m_vm(vm) {}
    
    ~Impl() = default;
    
    // Initialize the register allocator
    bool initialize() {
        // Get reference to physical register bank
        m_physicalRegisterBank = &m_vm->getRegisterBank();
        
        // Save initial physical register count
        m_initialRegisterCount = m_physicalRegisterBank->getNumRegisters();
        
        return true;
    }
    
    // Allocate virtual registers for a kernel
    bool allocateRegisters(uint32_t numPhysicalRegisters, 
                          uint32_t numWarps,
                          uint32_t threadsPerWarp) {
        // Check if we have a valid register bank
        if (!m_physicalRegisterBank) {
            return false;
        }
        
        // Store allocation parameters
        m_numPhysicalRegisters = numPhysicalRegisters;
        m_numWarps = numWarps;
        m_threadsPerWarp = threadsPerWarp;
        
        // Calculate total virtual registers needed
        m_totalVirtualRegisters = numWarps * threadsPerWarp * numPhysicalRegisters;
        
        // For now, we'll use a simple linear mapping
        // In real implementation, this would be more complex
        
        // Resize mapping table
        m_virtualToPhysicalMap.resize(m_totalVirtualRegisters);
        
        // Create mapping from virtual to physical registers
        for (uint32_t virtualReg = 0; virtualReg < m_totalVirtualRegisters; ++virtualReg) {
            // Simple round-robin mapping
            uint32_t physicalReg = virtualReg % numPhysicalRegisters;
            m_virtualToPhysicalMap[virtualReg] = physicalReg;
            
            // Track usage
            m_registerUsage[physicalReg].push_back(virtualReg);
        }
        
        // If we need more registers than available, create shadow storage
        if (numPhysicalRegisters < numWarps * threadsPerWarp) {
            // We'll need to spill some registers to memory
            m_spillSize = (m_totalVirtualRegisters - numPhysicalRegisters) * sizeof(uint64_t);
            
            // Allocate spill memory
            m_spillMemory = std::make_unique<uint64_t[]>(m_spillSize / sizeof(uint64_t));
            if (!m_spillMemory) {
                return false;
            }
            
            // Initialize spill memory tracking
            m_spillOffsets.resize(m_totalVirtualRegisters);
            for (uint32_t i = 0; i < m_totalVirtualRegisters; ++i) {
                if (i >= numPhysicalRegisters) {
                    m_spillOffsets[i] = (i - numPhysicalRegisters) * sizeof(uint64_t);
                } else {
                    m_spillOffsets[i] = 0;
                }
            }
        }
        
        // Set up virtual register banks for warps/threads
        m_registerBanks.resize(numWarps);
        for (uint32_t warpId = 0; warpId < numWarps; ++warpId) {
            m_registerBanks[warpId].resize(threadsPerWarp);
            
            for (uint32_t threadId = 0; threadId < threadsPerWarp; ++threadId) {
                // For now, just map directly to physical registers
                // In real implementation, this would use the mapping table
                m_registerBanks[warpId][threadId] = m_physicalRegisterBank;
            }
        }
        
        return true;
    }
    
    // Free register allocation for a kernel
    void freeRegisters() {
        // Reset all allocations
        m_numPhysicalRegisters = 0;
        m_numWarps = 0;
        m_threadsPerWarp = 0;
        m_totalVirtualRegisters = 0;
        m_spillSize = 0;
        
        // Clear mappings
        m_virtualToPhysicalMap.clear();
        m_spillOffsets.clear();
        m_registerBanks.clear();
        m_registerUsage.clear();
    }
    
    // Get number of physical registers available
    uint32_t getNumPhysicalRegisters() const {
        return m_numPhysicalRegisters;
    }
    
    // Get number of warps supported
    uint32_t getNumWarps() const {
        return m_numWarps;
    }
    
    // Get number of threads per warp
    uint32_t getThreadsPerWarp() const {
        return m_threadsPerWarp;
    }
    
    // Get total number of virtual registers allocated
    uint32_t getTotalVirtualRegisters() const {
        return m_totalVirtualRegisters;
    }
    
    // Get mapping from virtual to physical register
    uint32_t mapVirtualToPhysical(uint32_t virtualReg, uint32_t threadId) const {
        if (virtualReg >= m_totalVirtualRegisters) {
            return INVALID_REGISTER;
        }
        
        // Simple mapping based on current implementation
        // In real implementation, this would consider thread context
        return m_virtualToPhysicalMap[virtualReg];
    }
    
    // Get register bank for current context
    RegisterBank& getCurrentRegisterBank() {
        // For now, just return the physical register bank
        // In real implementation, this would depend on current warp/thread
        return *m_physicalRegisterBank;
    }
    
    // Save register state (for context switching)
    bool saveRegisterState(uint32_t warpId, uint32_t threadId) {
        if (warpId >= m_numWarps || threadId >= m_threadsPerWarp) {
            return false;
        }
        
        // For now, just return success
        // In real implementation, this would save register state to memory
        return true;
    }
    
    // Restore register state (for context switching)
    bool restoreRegisterState(uint32_t warpId, uint32_t threadId) {
        if (warpId >= m_numWarps || threadId >= m_threadsPerWarp) {
            return false;
        }
        
        // For now, just return success
        // In real implementation, this would restore register state from memory
        return true;
    }
    
    // Check if register is in use
    bool isRegisterInUse(uint32_t physicalReg) const {
        if (physicalReg >= m_numPhysicalRegisters) {
            return false;
        }
        
        // For now, assume all registers are in use
        // In real implementation, this would check actual usage
        return !m_registerUsage.at(physicalReg).empty();
    }
    
    // Get register usage statistics
    double getRegisterUtilization() const {
        if (m_numPhysicalRegisters == 0) {
            return 0.0;
        }
        
        // For now, calculate basic utilization
        uint32_t usedRegisters = 0;
        for (uint32_t reg = 0; reg < m_numPhysicalRegisters; ++reg) {
            if (!m_registerUsage.at(reg).empty()) {
                ++usedRegisters;
            }
        }
        
        return static_cast<double>(usedRegisters) / m_numPhysicalRegisters;
    }
    
private:
    // Core components
    PTXVM* m_vm;
    RegisterBank* m_physicalRegisterBank;  // Physical register bank
    
    // Allocation configuration
    uint32_t m_numPhysicalRegisters = 0;   // Number of physical registers
    uint32_t m_numWarps = 0;              // Number of warps
    uint32_t m_threadsPerWarp = 0;         // Threads per warp
    uint32_t m_totalVirtualRegisters = 0;  // Total virtual registers allocated
    size_t m_spillSize = 0;                // Size of spill memory needed
    
    // Register mappings
    std::vector<uint32_t> m_virtualToPhysicalMap;  // Virtual to physical register mapping
    std::vector<std::vector<RegisterBank*>> m_registerBanks;  // Register banks per warp/thread
    std::unordered_map<uint32_t, std::vector<uint32_t>> m_registerUsage;  // Usage tracking
    std::vector<size_t> m_spillOffsets;  // Offsets into spill memory
    std::unique_ptr<uint64_t[]> m_spillMemory;  // Spill memory for overflowed registers
    
    // Initial register count
    uint32_t m_initialRegisterCount = 0;
};

RegisterAllocator::RegisterAllocator(PTXVM* vm) : 
    pImpl(std::make_unique<Impl>(vm)) {}

RegisterAllocator::~RegisterAllocator() = default;

bool RegisterAllocator::initialize() {
    return pImpl->initialize();
}

bool RegisterAllocator::allocateRegisters(uint32_t numPhysicalRegisters, 
                                        uint32_t numWarps,
                                        uint32_t threadsPerWarp) {
    return pImpl->allocateRegisters(numPhysicalRegisters, numWarps, threadsPerWarp);
}

void RegisterAllocator::freeRegisters() {
    pImpl->freeRegisters();
}

uint32_t RegisterAllocator::getNumPhysicalRegisters() const {
    return pImpl->getNumPhysicalRegisters();
}

uint32_t RegisterAllocator::getNumWarps() const {
    return pImpl->getNumWarps();
}

uint32_t RegisterAllocator::getThreadsPerWarp() const {
    return pImpl->getThreadsPerWarp();
}

uint32_t RegisterAllocator::getTotalVirtualRegisters() const {
    return pImpl->getTotalVirtualRegisters();
}

uint32_t RegisterAllocator::mapVirtualToPhysical(uint32_t virtualReg, uint32_t threadId) const {
    return pImpl->mapVirtualToPhysical(virtualReg, threadId);
}

RegisterBank& RegisterAllocator::getCurrentRegisterBank() {
    return pImpl->getCurrentRegisterBank();
}

bool RegisterAllocator::saveRegisterState(uint32_t warpId, uint32_t threadId) {
    return pImpl->saveRegisterState(warpId, threadId);
}

bool RegisterAllocator::restoreRegisterState(uint32_t warpId, uint32_t threadId) {
    return pImpl->restoreRegisterState(warpId, threadId);
}

bool RegisterAllocator::isRegisterInUse(uint32_t physicalReg) const {
    return pImpl->isRegisterInUse(physicalReg);
}

double RegisterAllocator::getRegisterUtilization() const {
    return pImpl->getRegisterUtilization();
}