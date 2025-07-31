#include <gtest/gtest.h>
#include "memory/memory.hpp"
#include "memory/memory_optimizer.hpp"

// Test fixture for advanced memory operations
class AdvancedMemoryTest : public ::testing::Test {
protected:
    std::unique_ptr<MemorySubsystem> memory;
    
    void SetUp() override {
        memory = std::make_unique<MemorySubsystem>();
        ASSERT_TRUE(memory->initialize());
    }
};

// Helper function to fill memory with a pattern
void fillMemoryPattern(uint8_t* buffer, size_t size, uint8_t pattern) {
    for (size_t i = 0; i < size; ++i) {
        buffer[i] = pattern + static_cast<uint8_t>(i);
    }
}

// Helper function to verify memory pattern
bool verifyMemoryPattern(const uint8_t* buffer, size_t size, uint8_t pattern) {
    for (size_t i = 0; i < size; ++i) {
        if (buffer[i] != pattern + static_cast<uint8_t>(i)) {
            return false;
        }
    }
    return true;
}

// Test different memory spaces
TEST_F(AdvancedMemoryTest, TestMemorySpaces) {
    // Get pointers to different memory spaces
    void* globalMem = memory->getMemoryBuffer(MemorySpace::GLOBAL);
    void* sharedMem = memory->getMemoryBuffer(MemorySpace::SHARED);
    void* localMem = memory->getMemoryBuffer(MemorySpace::LOCAL);
    
    // Check that all memory spaces are accessible
    ASSERT_NE(globalMem, nullptr);
    ASSERT_NE(sharedMem, nullptr);
    ASSERT_NE(localMem, nullptr);
    
    // Verify they point to different regions
    EXPECT_NE(globalMem, sharedMem);
    EXPECT_NE(globalMem, localMem);
    EXPECT_NE(sharedMem, localMem);
    
    // Test writing to each memory space
    constexpr size_t TEST_SIZE = 1024;
    
    // Fill each memory space with different patterns
    fillMemoryPattern(static_cast<uint8_t*>(globalMem), TEST_SIZE, 0xA5);
    fillMemoryPattern(static_cast<uint8_t*>(sharedMem), TEST_SIZE, 0x5A);
    fillMemoryPattern(static_cast<uint8_t*>(localMem), TEST_SIZE, 0xFF);
    
    // Verify the contents of each memory space
    EXPECT_TRUE(verifyMemoryPattern(static_cast<uint8_t*>(globalMem), TEST_SIZE, 0xA5));
    EXPECT_TRUE(verifyMemoryPattern(static_cast<uint8_t*>(sharedMem), TEST_SIZE, 0x5A));
    EXPECT_TRUE(verifyMemoryPattern(static_cast<uint8_t*>(localMem), TEST_SIZE, 0xFF));
}

// Test memory access beyond allocated size
TEST_F(AdvancedMemoryTest, TestOutOfBoundsAccess) {
    // Get pointer to global memory
    void* globalMem = memory->getMemoryBuffer(MemorySpace::GLOBAL);
    ASSERT_NE(globalMem, nullptr);
    
    // Get the size of the global memory allocation
    MemoryAllocationInfo allocInfo = memory->getAllocationInfo(MemorySpace::GLOBAL);
    size_t memSize = allocInfo.size;
    
    // Should be able to access memory within bounds
    uint8_t* memPtr = static_cast<uint8_t*>(globalMem);
    
    // Access first byte - should be safe
    uint8_t originalValue = memPtr[0];
    memPtr[0] = 0xAA;
    EXPECT_EQ(memPtr[0], 0xAA);
    
    // Restore original value
    memPtr[0] = originalValue;
    
    // Try accessing one byte beyond allocated size - this should fail or not crash
    // Note: This is a basic test - in real implementation we would have better protection
    try {
        // Attempt out-of-bounds access
        volatile uint8_t value = memPtr[memSize];
        (void)value;  // Avoid unused warning
        
        // If we can read, we should also be able to write
        memPtr[memSize] = 0xFF;
        
        // If we reach here, check if the value was actually written
        // In some implementations, this might silently fail
        EXPECT_EQ(memPtr[memSize], 0xFF);
    } catch (...) {
        // We don't expect any exceptions, but this catches SEH on Windows
        FAIL() << "Out of bounds memory access caused an exception";
    }
}

// Test memory copy between different spaces
TEST_F(AdvancedMemoryTest, TestMemoryCopy) {
    const size_t TEST_SIZE = 256;
    const size_t SRC_OFFSET = 0;
    const size_t DST_OFFSET = 512;
    
    // Fill source buffer in global memory
    void* globalMem = memory->getMemoryBuffer(MemorySpace::GLOBAL);
    ASSERT_NE(globalMem, nullptr);
    uint8_t* globalMemPtr = static_cast<uint8_t*>(globalMem);
    
    // Fill with a pattern
    fillMemoryPattern(globalMemPtr + SRC_OFFSET, TEST_SIZE, 0x12);
    
    // Copy from global to shared memory
    void* sharedMem = memory->getMemoryBuffer(MemorySpace::SHARED);
    ASSERT_NE(sharedMem, nullptr);
    
    memory->copyMemory(MemorySpace::SHARED, DST_OFFSET,
                      MemorySpace::GLOBAL, SRC_OFFSET,
                      TEST_SIZE);
    
    // Verify shared memory contains the same pattern
    EXPECT_TRUE(verifyMemoryPattern(static_cast<uint8_t*>(sharedMem) + DST_OFFSET, 
                                   TEST_SIZE, 0x12));
    
    // Copy from shared back to global using different offset
    memory->copyMemory(MemorySpace::GLOBAL, SRC_OFFSET + TEST_SIZE,
                      MemorySpace::SHARED, DST_OFFSET,
                      TEST_SIZE);
    
    // Verify global memory contains the same pattern
    EXPECT_TRUE(verifyMemoryPattern(globalMemPtr + SRC_OFFSET + TEST_SIZE,
                                   TEST_SIZE, 0x12));
    
    // Verify original data still intact
    EXPECT_TRUE(verifyMemoryPattern(globalMemPtr + SRC_OFFSET,
                                   TEST_SIZE, 0x12));
}

// Test memory set operation
TEST_F(AdvancedMemoryTest, TestMemorySet) {
    const size_t TEST_SIZE = 128;
    const size_t OFFSET = 256;
    const uint8_t PATTERN = 0xA5;
    
    // Test setting memory in each space
    for (int space = 0; space < static_cast<int>(MemorySpace::MAX_MEMORY_SPACE); ++space) {
        MemorySpace memSpace = static_cast<MemorySpace>(space);
        
        // Skip parameter space as it might be small
        if (memSpace == MemorySpace::PARAMETER) {
            continue;
        }
        
        void* mem = memory->getMemoryBuffer(memSpace);
        if (!mem) {
            continue;  // Some memory spaces might not be implemented yet
        }
        
        // Set memory to pattern
        memory->setMemory(memSpace, OFFSET, PATTERN, TEST_SIZE);
        
        // Verify memory was set correctly
        const uint8_t* memPtr = static_cast<const uint8_t*>(mem);
        for (size_t i = 0; i < TEST_SIZE; ++i) {
            EXPECT_EQ(memPtr[OFFSET + i], PATTERN);
        }
        
        // Test zeroing memory
        memory->setMemory(memSpace, OFFSET, 0x00, TEST_SIZE);
        
        // Verify memory was zeroed
        for (size_t i = 0; i < TEST_SIZE; ++i) {
            EXPECT_EQ(memPtr[OFFSET + i], 0x00);
        }
    }
}

// Test large memory transfers
TEST_F(AdvancedMemoryTest, TestLargeMemoryTransfer) {
    const size_t LARGE_TEST_SIZE = 1024 * 1024;  // 1MB
    const size_t OFFSET = 0;
    
    // Test transferring in global memory space
    void* globalMem = memory->getMemoryBuffer(MemorySpace::GLOBAL);
    ASSERT_NE(globalMem, nullptr);
    
    // Fill source buffer with a pattern
    uint8_t* srcBuffer = new uint8_t[LARGE_TEST_SIZE];
    fillMemoryPattern(srcBuffer, LARGE_TEST_SIZE, 0x3C);
    
    // Write to memory
    memcpy(static_cast<uint8_t*>(globalMem) + OFFSET, srcBuffer, LARGE_TEST_SIZE);
    
    // Read back and verify
    const uint8_t* memPtr = static_cast<const uint8_t*>(globalMem);
    EXPECT_TRUE(verifyMemoryPattern(memPtr + OFFSET, LARGE_TEST_SIZE, 0x3C));
    
    // Modify some memory
    uint8_t* modifyPtr = static_cast<uint8_t*>(globalMem) + OFFSET + 512;
    *modifyPtr = 0xFF;
    
    // Test copying from global to shared memory
    void* sharedMem = memory->getMemoryBuffer(MemorySpace::SHARED);
    if (sharedMem) {
        // Copy from global to shared
        memory->copyMemory(MemorySpace::SHARED, OFFSET,
                          MemorySpace::GLOBAL, OFFSET,
                          LARGE_TEST_SIZE);
        
        // Verify shared memory contains the same pattern
        const uint8_t* sharedPtr = static_cast<const uint8_t*>(sharedMem);
        EXPECT_TRUE(verifyMemoryPattern(sharedPtr + OFFSET, LARGE_TEST_SIZE, 0x3C));
        
        // Verify modified byte was copied too
        EXPECT_EQ(sharedPtr[OFFSET + 512], 0xFF);
    }
    
    // Test copying from shared to local memory
    void* localMem = memory->getMemoryBuffer(MemorySpace::LOCAL);
    if (localMem) {
        // Copy from shared to local
        memory->copyMemory(MemorySpace::LOCAL, OFFSET,
                          MemorySpace::GLOBAL, OFFSET,
                          LARGE_TEST_SIZE);
        
        // Verify local memory contains the same pattern
        const uint8_t* localPtr = static_cast<const uint8_t*>(localMem);
        EXPECT_TRUE(verifyMemoryPattern(localPtr + OFFSET, LARGE_TEST_SIZE, 0x3C));
        
        // Verify modified byte was copied too
        EXPECT_EQ(localPtr[OFFSET + 512], 0xFF);
    }
    
    delete[] srcBuffer;
}

// Test memory access alignment
TEST_F(AdvancedMemoryTest, TestMemoryAccessAlignment) {
    const size_t BUFFER_SIZE = 1024;
    const size_t BASE_OFFSET = 0;
    
    // Get global memory
    void* globalMem = memory->getMemoryBuffer(MemorySpace::GLOBAL);
    ASSERT_NE(globalMem, nullptr);
    uint8_t* memPtr = static_cast<uint8_t*>(globalMem);
    
    // Fill with known pattern
    for (size_t i = 0; i < BUFFER_SIZE; ++i) {
        memPtr[BASE_OFFSET + i] = static_cast<uint8_t>(i);
    }
    
    // Test aligned accesses
    #define CHECK_ALIGNED_ACCESS(type, typeSize) \
    { \
        for (size_t offset = 0; offset < BUFFER_SIZE - sizeof(type); offset += typeSize) { \
            type* ptr = reinterpret_cast<type*>(memPtr + BASE_OFFSET + offset); \
            type value = *ptr; \
            \
            // Store back with different value
            *ptr = static_cast<type>(value + 1); \
            \
            // Check that only the target bytes were modified
            for (size_t i = 0; i < sizeof(type); ++i) {
                EXPECT_EQ(memPtr[BASE_OFFSET + offset + i], reinterpret_cast<uint8_t*>(&value)[i] + 1);
            }
        } \
    }
    
    // Test various data types
    CHECK_ALIGNED_ACCESS(uint8_t, 1)  // 1-byte access
    CHECK_ALIGNED_ACCESS(uint16_t, 2)  // 2-byte access
    CHECK_ALIGNED_ACCESS(uint32_t, 4)  // 4-byte access
    CHECK_ALIGNED_ACCESS(uint64_t, 8)  // 8-byte access
    
    #undef CHECK_ALIGNED_ACCESS
}

// Test unaligned memory access
TEST_F(AdvancedMemoryTest, TestUnalignedMemoryAccess) {
    const size_t BUFFER_SIZE = 1024;
    const size_t BASE_OFFSET = 0;
    
    // Get global memory
    void* globalMem = memory->getMemoryBuffer(MemorySpace::GLOBAL);
    ASSERT_NE(globalMem, nullptr);
    uint8_t* memPtr = static_cast<uint8_t*>(globalMem);
    
    // Fill with known pattern
    for (size_t i = 0; i < BUFFER_SIZE; ++i) {
        memPtr[BASE_OFFSET + i] = static_cast<uint8_t>(i);
    }
    
    // Test unaligned 32-bit accesses
    for (size_t offset = 1; offset < BUFFER_SIZE - sizeof(uint32_t); offset += 2) {
        // Read through unaligned pointer
        uint32_t* unalignedPtr = reinterpret_cast<uint32_t*>(memPtr + BASE_OFFSET + offset);
        uint32_t value = *unalignedPtr;
        
        // Store back at next unaligned location
        uint32_t* unalignedStorePtr = reinterpret_cast<uint32_t*>(memPtr + BASE_OFFSET + offset + 1);
        *unalignedStorePtr = value;
        
        // Verify memory wasn't corrupted
        for (size_t i = 0; i < sizeof(uint32_t); ++i) {
            // Ensure we didn't corrupt previous bytes
            if (offset + i > 0) {
                EXPECT_EQ(memPtr[BASE_OFFSET + offset - 1 + i], static_cast<uint8_t>((offset - 1 + i) & 0xFF));
            }
            
            // Ensure current bytes match expected
            EXPECT_EQ(memPtr[BASE_OFFSET + offset + i], reinterpret_cast<uint8_t*>(&value)[i]);
        }
    }
}

// Test memory barriers
TEST_F(AdvancedMemoryTest, TestMemoryBarriers) {
    // This is a simplified test for memory barrier functionality
    // Actual implementation would involve multiple threads
    
    // Initialize memory
    void* globalMem = memory->getMemoryBuffer(MemorySpace::GLOBAL);
    ASSERT_NE(globalMem, nullptr);
    
    // Test pre-barrier state
    uint8_t* memPtr = static_cast<uint8_t*>(globalMem);
    memPtr[0] = 0x01;
    memPtr[1] = 0x02;
    
    // Apply a memory barrier
    memory->memoryBarrier();
    
    // After barrier, memory should be consistent
    EXPECT_EQ(memPtr[0], 0x01);
    EXPECT_EQ(memPtr[1], 0x02);
    
    // For now, the barrier just ensures sequential consistency
    // A full test would involve multiple threads and ordering constraints
}

// Test memory allocation information
TEST_F(AdvancedMemoryTest, TestMemoryAllocationInfo) {
    // Test each memory space
    for (int space = 0; space < static_cast<int>(MemorySpace::MAX_MEMORY_SPACE); ++space) {
        MemorySpace memSpace = static_cast<MemorySpace>(space);
        
        // Get allocation info for this space
        MemoryAllocationInfo info = memory->getAllocationInfo(memSpace);
        
        // Check that base address is valid
        if (info.flags & ALLOCATION_FLAG_READABLE) {
            // Memory should be readable
            const uint8_t* memPtr = static_cast<const uint8_t*>(info.baseAddress);
            if (memPtr && info.size >= 1) {
                // Save original value
                uint8_t originalValue = memPtr[0];
                
                // Try reading
                uint8_t value = memPtr[0];
                
                // Try writing if writable
                if (info.flags & ALLOCATION_FLAG_WRITABLE) {
                    uint8_t newValue = ~value;
                    const_cast<uint8_t*>(memPtr)[0] = newValue;
                    
                    // Restore original value
                    const_cast<uint8_t*>(memPtr)[0] = originalValue;
                }
            }
        }
    }
}

// Test memory statistics
TEST_F(AdvancedMemoryTest, TestMemoryStatistics) {
    // Get initial statistics
    MemoryAllocationInfo globalInfo = memory->getAllocationInfo(MemorySpace::GLOBAL);
    MemoryAllocationInfo sharedInfo = memory->getAllocationInfo(MemorySpace::SHARED);
    MemoryAllocationInfo localInfo = memory->getAllocationInfo(MemorySpace::LOCAL);
    
    // Record initial sizes
    size_t globalSize = globalInfo.size;
    size_t sharedSize = sharedInfo.size;
    size_t localSize = localInfo.size;
    
    // These are simple sanity checks
    // More detailed validation would depend on implementation
    EXPECT_GT(globalSize, 0u);
    EXPECT_GT(sharedSize, 0u);
    EXPECT_GT(localSize, 0u);
    
    // Test that sizes are consistent
    if (globalInfo.baseAddress <= sharedInfo.baseAddress && 
        sharedInfo.baseAddress < reinterpret_cast<void*>(reinterpret_cast<uint8_t*>(globalInfo.baseAddress) + globalSize)) {
        // Shared memory is within global memory region
        // This could be valid depending on implementation
        EXPECT_LE(reinterpret_cast<uint8_t*>(sharedInfo.baseAddress) - reinterpret_cast<uint8_t*>(globalInfo.baseAddress), globalSize);
    }
    
    if (globalInfo.baseAddress <= localInfo.baseAddress && 
        localInfo.baseAddress < reinterpret_cast<void*>(reinterpret_cast<uint8_t*>(globalInfo.baseAddress) + globalSize)) {
        // Local memory is within global memory region
        // This could be valid depending on implementation
        EXPECT_LE(reinterpret_cast<uint8_t*>(localInfo.baseAddress) - reinterpret_cast<uint8_t*>(globalInfo.baseAddress), globalSize);
    }
}

// Test memory synchronization
TEST_F(AdvancedMemoryTest, TestMemorySynchronization) {
    // This is a simplified test for memory synchronization
    // Actual implementation would involve multiple threads
    
    // Initialize memory
    void* globalMem = memory->getMemoryBuffer(MemorySpace::GLOBAL);
    ASSERT_NE(globalMem, nullptr);
    
    // Fill memory with a pattern
    uint8_t* memPtr = static_cast<uint8_t*>(globalMem);
    for (size_t i = 0; i < 1024; ++i) {
        memPtr[i] = static_cast<uint8_t>(i % 256);
    }
    
    // Synchronize memory
    memory->synchronizeMemory(MemorySpace::GLOBAL, 0, 1024);
    
    // For now, just verify that the values are unchanged after synchronization
    for (size_t i = 0; i < 1024; ++i) {
        EXPECT_EQ(memPtr[i], static_cast<uint8_t>(i % 256));
    }
    
    // This is a placeholder for more complex tests that would involve
    // device-host synchronization in a real implementation
}

// Test memory access flags
TEST_F(AdvancedMemoryTest, TestMemoryAccessFlags) {
    // Test each memory space
    for (int space = 0; space < static_cast<int>(MemorySpace::MAX_MEMORY_SPACE); ++space) {
        MemorySpace memSpace = static_cast<MemorySpace>(space);
        
        // Get allocation info
        MemoryAllocationInfo info = memory->getAllocationInfo(memSpace);
        
        // Check that flags make sense
        if (info.flags & ALLOCATION_FLAG_READABLE) {
            // Memory should be readable
            const uint8_t* memPtr = static_cast<const uint8_t*>(info.baseAddress);
            if (memPtr && info.size > 0) {
                // Read first byte to verify readability
                [[maybe_unused]] uint8_t value = memPtr[0];
                // No assertion here - this is what we're testing
            }
        }
        
        if (info.flags & ALLOCATION_FLAG_WRITABLE) {
            // Memory should be writable
            uint8_t* memPtr = static_cast<uint8_t*>(info.baseAddress);
            if (memPtr && info.size > 0) {
                // Save original value
                uint8_t originalValue = memPtr[0];
                
                // Try writing
                memPtr[0] = ~originalValue;
                
                // Restore original value
                memPtr[0] = originalValue;
            }
        }
        
        if (info.flags & ALLOCATION_FLAG_EXECUTABLE) {
            // Memory should be executable
            // This is hard to test without actual code generation
            // Just verify we can get the pointer
            void* execPtr = info.baseAddress;
            if (execPtr) {
                // In real implementation, we would generate and execute code here
                // This is just a placeholder
                EXPECT_TRUE(true);  // Just to use the variable
            }
        }
    }
}

// Test TLB functionality
TEST(MemorySubsystemTest, TlbFunctionality) {
    MemorySubsystem memory;
    ASSERT_TRUE(memory.initialize());
    
    // Initialize TLB with 32 entries
    memory.initializeTLB(32);
    
    // Test address translation
    uint64_t virtualAddress = 0x1000;
    MemoryAccessResult result = memory.translateAddress(virtualAddress);
    
    // Should result in a page fault since page is not mapped
    EXPECT_TRUE(result.pageFault);
    EXPECT_FALSE(result.success);
    
    // Handle page fault
    ASSERT_TRUE(memory.handlePageFault(virtualAddress));
    
    // Try translation again
    result = memory.translateAddress(virtualAddress);
    
    // Should now succeed
    EXPECT_FALSE(result.pageFault);
    EXPECT_TRUE(result.success);
    
    // Check TLB statistics
    EXPECT_EQ(memory.getTlbMisses(), 1);
    EXPECT_EQ(memory.getPageFaults(), 1);
    
    // Access same address again (should be TLB hit)
    result = memory.translateAddress(virtualAddress);
    EXPECT_FALSE(result.pageFault);
    EXPECT_TRUE(result.success);
    
    // Check TLB statistics
    EXPECT_EQ(memory.getTlbHits(), 1);
}

// Test cache configuration
TEST(MemorySubsystemTest, CacheConfiguration) {
    MemorySubsystem memory;
    ASSERT_TRUE(memory.initialize());
    
    // Configure cache
    CacheConfig config;
    config.cacheSize = 32 * 1024;  // 32KB
    config.lineSize = 64;          // 64 bytes
    config.associativity = 4;      // 4-way associative
    config.writeThrough = false;
    config.replacementPolicy = 0;  // LRU
    
    memory.configureCache(config);
    
    // Test cache simulation with some addresses
    std::vector<uint64_t> addresses = {0x1000, 0x1004, 0x1008, 0x100C};
    
    // Calculate coalescing efficiency
    size_t efficiency = memory.calculateCoalescingEfficiency(addresses);
    EXPECT_GT(efficiency, 0);
    
    // Check if accesses are coalesced
    bool coalesced = memory.isAccessCoalesced(addresses);
    EXPECT_TRUE(coalesced);
}

// Test shared memory bank conflicts
TEST(MemorySubsystemTest, SharedMemoryBankConflicts) {
    MemorySubsystem memory;
    ASSERT_TRUE(memory.initialize());
    
    // Configure shared memory
    SharedMemoryConfig config;
    config.bankCount = 32;
    config.bankWidth = 4;
    config.bankSize = 1024;
    
    memory.configureSharedMemory(config);
    
    // Test bank conflict detection
    // These addresses should cause conflicts with 32 banks of 4 bytes each
    std::vector<uint64_t> addresses = {0, 128, 256, 384}; // All map to bank 0
    
    size_t conflicts = memory.getBankConflicts(addresses);
    EXPECT_EQ(conflicts, 3); // 3 conflicts (first access doesn't conflict)
    
    // Test with non-conflicting addresses
    std::vector<uint64_t> nonConflicting = {0, 4, 8, 12}; // Map to banks 0, 1, 2, 3
    
    conflicts = memory.getBankConflicts(nonConflicting);
    EXPECT_EQ(conflicts, 0); // No conflicts
}

// Test memory optimizer framework
TEST(MemoryOptimizerTest, BasicFunctionality) {
    MemoryOptimizerFramework optimizer;
    
    // Test default configuration
    MemoryOptimizerConfig config = optimizer.getConfig();
    EXPECT_FALSE(config.enableCaching);
    EXPECT_FALSE(config.enableSharedMemoryOptimization);
    EXPECT_FALSE(config.enableTLB);
    
    // Configure optimizer
    MemoryOptimizerConfig newConfig;
    newConfig.enableCaching = true;
    newConfig.cacheSize = 64 * 1024;  // 64KB
    newConfig.cacheLineSize = 64;
    newConfig.cacheAssociativity = 8;
    newConfig.writeThrough = false;
    
    newConfig.enableSharedMemoryOptimization = true;
    newConfig.sharedMemoryBanks = 32;
    newConfig.sharedMemoryBankWidth = 4;
    
    newConfig.enableTLB = true;
    newConfig.tlbEntries = 64;
    newConfig.pageSize = 4096;
    
    newConfig.optimizationLevel = 2;
    
    optimizer.configure(newConfig);
    
    config = optimizer.getConfig();
    EXPECT_TRUE(config.enableCaching);
    EXPECT_TRUE(config.enableSharedMemoryOptimization);
    EXPECT_TRUE(config.enableTLB);
    EXPECT_EQ(config.optimizationLevel, 2);
}

// Test memory access pattern analysis
TEST(MemoryOptimizerTest, AccessPatternAnalysis) {
    MemoryOptimizerFramework optimizer;
    
    // Test with coalesced access pattern
    std::vector<uint64_t> coalescedAddresses;
    for (int i = 0; i < 32; i++) {
        coalescedAddresses.push_back(i * 4); // Consecutive 4-byte accesses
    }
    
    optimizer.analyzeAccessPattern(coalescedAddresses);
    
    // Check coalescing efficiency
    double efficiency = optimizer.getCoalescingEfficiency();
    EXPECT_GT(efficiency, 80.0); // Should be highly efficient
    
    // Test with random access pattern
    std::vector<uint64_t> randomAddresses = {0x1000, 0x5000, 0x3000, 0x7000, 0x2000};
    
    optimizer.analyzeAccessPattern(randomAddresses);
    
    // Get statistics
    size_t bankConflicts = optimizer.getBankConflicts();
    double cacheHitRate = optimizer.getCacheHitRate();
    double tlbHitRate = optimizer.getTlbHitRate();
    
    // These are just basic checks - values will depend on internal simulation
    EXPECT_GE(bankConflicts, 0);
    EXPECT_GE(cacheHitRate, 0.0);
    EXPECT_LE(cacheHitRate, 100.0);
    EXPECT_GE(tlbHitRate, 0.0);
    EXPECT_LE(tlbHitRate, 100.0);
}

// Test access pattern optimization
TEST(MemoryOptimizerTest, AccessPatternOptimization) {
    MemoryOptimizerFramework optimizer;
    
    // Configure for advanced optimization
    MemoryOptimizerConfig config;
    config.optimizationLevel = 2;
    optimizer.configure(config);
    
    // Test with unsorted addresses
    std::vector<uint64_t> unsorted = {0x5000, 0x1000, 0x3000, 0x2000, 0x4000};
    std::vector<uint64_t> optimized = optimizer.optimizeAccessPattern(unsorted);
    
    // Check that optimized version is sorted
    EXPECT_TRUE(std::is_sorted(optimized.begin(), optimized.end()));
    
    // Check that size is preserved (no duplicates in this case)
    EXPECT_EQ(optimized.size(), unsorted.size());
    
    // Test with duplicate addresses
    std::vector<uint64_t> withDuplicates = {0x1000, 0x2000, 0x1000, 0x3000, 0x2000};
    std::vector<uint64_t> optimizedDuplicates = optimizer.optimizeAccessPattern(withDuplicates);
    
    // Check that duplicates are removed
    EXPECT_EQ(optimizedDuplicates.size(), 3);
    EXPECT_TRUE(std::is_sorted(optimizedDuplicates.begin(), optimizedDuplicates.end()));
}
