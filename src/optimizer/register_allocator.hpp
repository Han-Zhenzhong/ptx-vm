#ifndef REGISTER_ALLOCATOR_HPP
#define REGISTER_ALLOCATOR_HPP

#include <cstdint>
#include <vector>
#include <unordered_map>
#include <memory>
#include "registers/register_bank.hpp"

// Forward declarations
class PTXVM;

class RegisterAllocator {
public:
    // Constructor/destructor
    explicit RegisterAllocator(PTXVM* vm);
    ~RegisterAllocator();

    // Initialize the register allocator
    bool initialize();

    // Allocate virtual registers for a kernel
    bool allocateRegisters(uint32_t numPhysicalRegisters, 
                          uint32_t numWarps,
                          uint32_t threadsPerWarp);

    // Free register allocation for a kernel
    void freeRegisters();

    // Get number of physical registers available
    uint32_t getNumPhysicalRegisters() const;

    // Get number of warps supported
    uint32_t getNumWarps() const;

    // Get number of threads per warp
    uint32_t getThreadsPerWarp() const;

    // Get total number of virtual registers allocated
    uint32_t getTotalVirtualRegisters() const;

    // Get mapping from virtual to physical register
    uint32_t mapVirtualToPhysical(uint32_t virtualReg, uint32_t threadId = 0) const;

    // Get register bank for current context
    RegisterBank& getCurrentRegisterBank();

    // Save register state (for context switching)
    bool saveRegisterState(uint32_t warpId, uint32_t threadId);

    // Restore register state (for context switching)
    bool restoreRegisterState(uint32_t warpId, uint32_t threadId);

    // Check if register is in use
    bool isRegisterInUse(uint32_t physicalReg) const;

    // Get register usage statistics
    double getRegisterUtilization() const;

private:
    // Private implementation details
    class Impl;
    std::unique_ptr<Impl> pImpl;
};

#endif // REGISTER_ALLOCATOR_HPP