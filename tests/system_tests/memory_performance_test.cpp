#include "gtest/gtest.h"
#include "vm.hpp"
#include "memory_optimizer.hpp"
#include "performance_counters.hpp"

// Memory performance test fixture
class MemoryPerformanceTest : public ::testing::Test {
protected:
    std::unique_ptr<VM> m_vm;
    std::unique_ptr<MemoryOptimizerFramework> m_memoryOptimizer;
    
    void SetUp() override {
        // Initialize VM components
        m_vm = std::make_unique<VM>();
        ASSERT_TRUE(m_vm->initialize());
        ASSERT_TRUE(m_loader->initialize());
        
        // Initialize memory optimizer
        m_memoryOptimizer = std::make_unique<MemoryOptimizerFramework>();
        
        // Configure memory optimizer
        MemoryOptimizerConfig config;
        config.enableCaching = true;
        config.cacheSize = 256 * 1024;  // 256KB
        config.cacheLineSize = 64;         // 64 bytes
        config.cacheAssociativity = 8;       // 8-way associative
        config.enableSharedMemoryOptimization = true;
        config.sharedMemoryBanks = 32;       // 32 banks
        config.sharedMemoryBankWidth = 4;     // 4 bytes per bank
        config.enableTLB = true;
        config.tlbEntries = 64;              // 64 TLB entries
        config.pageSize = 4096;             // 4KB pages
        config.optimizationLevel = 2;         // Advanced optimization
        
        m_memoryOptimizer->configure(config);
        
        // Set memory optimizer in VM
        m_vm->setMemoryOptimizer(m_memoryOptimizer.get());
    }
    
    void TearDown() override {
        // Clean up
        m_vm.reset();
        m_loader.reset();
        m_memoryOptimizer.reset();
    }
};

// Test memory optimization performance with coalesced access
TEST_F(MemoryPerformanceTest, CoalescedAccessPerformance) {
    // Load test program
    ASSERT_TRUE(m_loader->loadBinary("../examples/coalesced_memory_example.ptx"));
    
    // Get kernel info
    KernelInfo info;
    ASSERT_TRUE(m_loader->getKernelInfo("coalesced_kernel", info));
    
    // Set execution parameters
    m_vm->setGridSize(1);
    m_vm->setBlockSize(info.maxThreadsPerBlock);
    
    // Load kernel
    ASSERT_TRUE(m_vm->loadKernel("coalesced_kernel"));
    
    // Run performance test
    PerformanceCounters counters;
    ASSERT_TRUE(m_vm->runKernel(&counters));
    
    // Verify results
    EXPECT_GT(counters.global_memory_reads, 0);
    EXPECT_GT(counters.cache_hits, 0);
    
    // Calculate performance metrics
    double cacheHitRate = static_cast<double>(counters.cache_hits) / 
                          (counters.cache_hits + counters.cache_misses);
    
    // Ensure cache hit rate is above 85%
    EXPECT_GT(cacheHitRate, 0.85);
    
    // Print performance metrics
    std::cout << "IPC: " << counters.instructions_executed / static_cast<double>(counters.total_cycles) << std::endl;
    std::cout << "Cache hit rate: " << cacheHitRate * 100 << "%" << std::endl;
    std::cout << "Memory bandwidth: " << counters.memory_bandwidth / (1024.0 * 1024.0 * 1024.0) << " GB/s" << std::endl;
}

// Test memory optimization performance with strided access
TEST_F(MemoryPerformanceTest, StridedAccessPerformance) {
    // Load test program
    ASSERT_TRUE(m_loader->loadBinary("../examples/strided_memory_example.ptx"));
    
    // Get kernel info
    KernelInfo info;
    ASSERT_TRUE(m_loader->getKernelInfo("strided_kernel", info));
    
    // Set execution parameters
    m_vm->setGridSize(1);
    m_vm->setBlockSize(info.maxThreadsPerBlock);
    
    // Load kernel
    ASSERT_TRUE(m_vm->loadKernel("strided_kernel"));
    
    // Run performance test
    PerformanceCounters counters;
    ASSERT_TRUE(m_vm->runKernel(&counters));
    
    // Verify results
    EXPECT_GT(counters.global_memory_reads, 0);
    EXPECT_GT(counters.cache_hits, 0);
    
    // Calculate performance metrics
    double cacheHitRate = static_cast<double>(counters.cache_hits) / 
                          (counters.cache_hits + counters.cache_misses);
    
    // Ensure cache hit rate is above 80%
    EXPECT_GT(cacheHitRate, 0.80);
    
    // Print performance metrics
    std::cout << "IPC: " << counters.instructions_executed / static_cast<double>(counters.total_cycles) << std::endl;
    std::cout << "Cache hit rate: " << cacheHitRate * 100 << "%" << std::endl;
    std::cout << "Memory bandwidth: " << counters.memory_bandwidth / (1024.0 * 1024.0 * 1024.0) << " GB/s" << std::endl;
}

// Test memory optimization performance with shared memory
TEST_F(MemoryPerformanceTest, SharedMemoryPerformance) {
    // Load test program
    ASSERT_TRUE(m_loader->loadBinary("../examples/shared_memory_example.ptx"));
    
    // Get kernel info
    KernelInfo info;
    ASSERT_TRUE(m_loader->getKernelInfo("shared_memory_kernel", info));
    
    // Set execution parameters
    m_vm->setGridSize(1);
    m_vm->setBlockSize(info.maxThreadsPerBlock);
    
    // Load kernel
    ASSERT_TRUE(m_vm->loadKernel("shared_memory_kernel"));
    
    // Run performance test
    PerformanceCounters counters;
    ASSERT_TRUE(m_vm->runKernel(&counters));
    
    // Verify results
    EXPECT_GT(counters.shared_memory_reads, 0);
    EXPECT_EQ(counters.bank_conflicts, 0);
    
    // Print performance metrics
    std::cout << "IPC: " << counters.instructions_executed / static_cast<double>(counters.total_cycles) << std::endl;
    std::cout << "Shared memory reads: " << counters.shared_memory_reads << std::endl;
    std::cout << "Bank conflicts: " << counters.bank_conflicts << std::endl;
    std::cout << "Memory bandwidth: " << counters.memory_bandwidth / (1024.0 * 1024.0 * 1024.0) << " GB/s" << std::endl;
}

// Test memory optimization performance with TLB
TEST_F(MemoryPerformanceTest, TLBPerformance) {
    // Load test program
    ASSERT_TRUE(m_loader->loadBinary("../examples/tlb_memory_example.ptx"));
    
    // Get kernel info
    KernelInfo info;
    ASSERT_TRUE(m_loader->getKernelInfo("tlb_kernel", info));
    
    // Set execution parameters
    m_vm->setGridSize(1);
    m_vm->setBlockSize(info.maxThreadsPerBlock);
    
    // Load kernel
    ASSERT_TRUE(m_vm->loadKernel("tlb_kernel"));
    
    // Run performance test
    PerformanceCounters counters;
    ASSERT_TRUE(m_vm->runKernel(&counters));
    
    // Verify results
    EXPECT_GT(counters.global_memory_reads, 0);
    EXPECT_GT(counters.page_faults, 0);
    
    // Calculate performance metrics
    double tlbHitRate = static_cast<double>(counters.tlb_hits) / 
                         (counters.tlb_hits + counters.tlb_misses);
    
    // Ensure TLB hit rate is above 90%
    EXPECT_GT(tlbHitRate, 0.90);
    
    // Print performance metrics
    std::cout << "IPC: " << counters.instructions_executed / static_cast<double>(counters.total_cycles) << std::endl;
    std::cout << "TLB hit rate: " << tlbHitRate * 100 << "%" << std::endl;
    std::cout << "Page faults: " << counters.page_faults << std::endl;
    std::cout << "Memory bandwidth: " << counters.memory_bandwidth / (1024.0 * 1024.0 * 1024.0) << " GB/s" << std::endl;
}

// Test overall memory optimization performance
TEST_F(MemoryPerformanceTest, OverallMemoryPerformance) {
    // Configure full optimization
    MemoryOptimizerConfig config;
    config.enableCaching = true;
    config.cacheSize = 256 * 1024;  // 256KB
    config.cacheLineSize = 64;         // 64 bytes
    config.cacheAssociativity = 8;       // 8-way associative
    config.enableSharedMemoryOptimization = true;
    config.sharedMemoryBanks = 32;       // 32 banks
    config.sharedMemoryBankWidth = 4;     // 4 bytes per bank
    config.enableTLB = true;
    config.tlbEntries = 64;              // 64 TLB entries
    config.pageSize = 4096;             // 4KB pages
    config.optimizationLevel = 2;         // Advanced optimization
    
    m_memoryOptimizer->configure(config);
    
    // Load test program
    ASSERT_TRUE(m_loader->loadBinary("../examples/complex_memory_example.ptx"));
    
    // Get kernel info
    KernelInfo info;
    ASSERT_TRUE(m_loader->getKernelInfo("complex_kernel", info));
    
    // Set execution parameters
    m_vm->setGridSize(1);
    m_vm->setBlockSize(info.maxThreadsPerBlock);
    
    // Load kernel
    ASSERT_TRUE(m_vm->loadKernel("complex_kernel"));
    
    // Run performance test
    PerformanceCounters counters;
    ASSERT_TRUE(m_vm->runKernel(&counters));
    
    // Verify results
    EXPECT_GT(counters.instructions_executed, 0);
    EXPECT_GT(counters.memory_bandwidth, 0);
    
    // Calculate performance metrics
    double cacheHitRate = static_cast<double>(counters.cache_hits) / 
                          (counters.cache_hits + counters.cache_misses);
    
    double tlbHitRate = static_cast<double>(counters.tlb_hits) / 
                        (counters.tlb_hits + counters.tlb_misses);
    
    // Ensure minimum performance standards
    EXPECT_GT(counters.instructions_per_second / 1e9, 1.0);  // At least 1 billion instructions per second
    EXPECT_GT(cacheHitRate, 0.85);
    EXPECT_GT(tlbHitRate, 0.90);
    
    // Print performance metrics
    std::cout << "IPC: " << counters.instructions_executed / static_cast<double>(counters.total_cycles) << std::endl;
    std::cout << "Instructions per second: " << counters.instructions_per_second / 1e9 << " GIPS" << std::endl;
    std::cout << "Cache hit rate: " << cacheHitRate * 100 << "%" << std::endl;
    std::cout << "TLB hit rate: " << tlbHitRate * 100 << "%" << std::endl;
    std::cout << "Memory bandwidth: " << counters.memory_bandwidth / (1024.0 * 1024.0 * 1024.0) << " GB/s" << std::endl;
    std::cout << "Execution time: " << counters.execution_time << " ms" << std::endl;
}

// Test memory optimization with different configurations
TEST_F(MemoryPerformanceTest, OptimizationConfigurationComparison) {
    // Base configuration
    MemoryOptimizerConfig baseConfig;
    baseConfig.enableCaching = true;
    baseConfig.cacheSize = 64 * 1024;  // 64KB
    baseConfig.cacheLineSize = 32;        // 32 bytes
    baseConfig.cacheAssociativity = 4;      // 4-way associative
    baseConfig.enableSharedMemoryOptimization = true;
    baseConfig.sharedMemoryBanks = 16;     // 16 banks
    baseConfig.sharedMemoryBankWidth = 4;   // 4 bytes per bank
    baseConfig.enableTLB = true;
    baseConfig.tlbEntries = 32;           // 32 TLB entries
    baseConfig.pageSize = 4096;           // 4KB pages
    baseConfig.optimizationLevel = 1;       // Basic optimization
    
    // Advanced configuration
    MemoryOptimizerConfig advancedConfig;
    advancedConfig.enableCaching = true;
    advancedConfig.cacheSize = 256 * 1024;  // 256KB
    advancedConfig.cacheLineSize = 64;         // 64 bytes
    advancedConfig.cacheAssociativity = 8;       // 8-way associative
    advancedConfig.enableSharedMemoryOptimization = true;
    advancedConfig.sharedMemoryBanks = 32;       // 32 banks
    advancedConfig.sharedMemoryBankWidth = 4;     // 4 bytes per bank
    advancedConfig.enableTLB = true;
    advancedConfig.tlbEntries = 64;              // 64 TLB entries
    advancedConfig.pageSize = 4096;             // 4KB pages
    advancedConfig.optimizationLevel = 2;         // Advanced optimization
    
    // Load test program
    ASSERT_TRUE(m_loader->loadBinary("../examples/complex_memory_example.ptx"));
    
    // Get kernel info
    KernelInfo info;
    ASSERT_TRUE(m_loader->getKernelInfo("complex_kernel", info));
    
    // Set execution parameters
    m_vm->setGridSize(1);
    m_vm->setBlockSize(info.maxThreadsPerBlock);
    
    // Run base configuration
    m_memoryOptimizer->configure(baseConfig);
    PerformanceCounters baseCounters;
    ASSERT_TRUE(m_vm->loadKernel("complex_kernel"));
    ASSERT_TRUE(m_vm->runKernel(&baseCounters));
    
    // Run advanced configuration
    m_memoryOptimizer->configure(advancedConfig);
    PerformanceCounters advancedCounters;
    ASSERT_TRUE(m_vm->loadKernel("complex_kernel"));
    ASSERT_TRUE(m_vm->runKernel(&advancedCounters));
    
    // Calculate performance metrics
    double baseCacheHitRate = static_cast<double>(baseCounters.cache_hits) / 
                            (baseCounters.cache_hits + baseCounters.cache_misses);
    
    double advancedCacheHitRate = static_cast<double>(advancedCounters.cache_hits) / 
                               (advancedCounters.cache_hits + advancedCounters.cache_misses);
    
    double baseTLBHitRate = static_cast<double>(baseCounters.tlb_hits) / 
                            (baseCounters.tlb_hits + baseCounters.tlb_misses);
    
    double advancedTLBHitRate = static_cast<double>(advancedCounters.tlb_hits) / 
                               (advancedCounters.tlb_hits + advancedCounters.tlb_misses);
    
    // Compare configurations
    std::cout << "Base Configuration" << std::endl;
    std::cout << "Cache hit rate: " << baseCacheHitRate * 100 << "%" << std::endl;
    std::cout << "TLB hit rate: " << baseTLBHitRate * 100 << "%" << std::endl;
    std::cout << "Execution time: " << baseCounters.execution_time << " ms" << std::endl;
    std::cout << "Memory bandwidth: " << baseCounters.memory_bandwidth / (1024.0 * 1024.0 * 1024.0) << " GB/s" << std::endl;
    
    std::cout << "\nAdvanced Configuration" << std::endl;
    std::cout << "Cache hit rate: " << advancedCacheHitRate * 100 << "%" << std::endl;
    std::cout << "TLB hit rate: " << advancedTLBHitRate * 100 << "%" << std::endl;
    std::cout << "Execution time: " << advancedCounters.execution_time << " ms" << std::endl;
    std::cout << "Memory bandwidth: " << advancedCounters.memory_bandwidth / (1024.0 * 1024.0 * 1024.0) << " GB/s" << std::endl;
    
    // Ensure advanced configuration provides better performance
    EXPECT_GT(advancedCounters.memory_bandwidth, baseCounters.memory_bandwidth);
    EXPECT_LT(advancedCounters.execution_time, baseCounters.execution_time);
    
    // Ensure advanced configuration has better cache hit rate
    EXPECT_GT(advancedCacheHitRate, baseCacheHitRate);
    
    // Ensure advanced configuration has better TLB hit rate
    EXPECT_GT(advancedTLBHitRate, baseTLBHitRate);
}