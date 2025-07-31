#include <gtest/gtest.h>
#include <benchmark/benchmark.h>
#include "vm.hpp"
#include "optimizer/register_allocator.hpp"

// Benchmark different register allocation scenarios
static void BM_RegisterAllocation_Simple(benchmark::State& state) {
    // Setup - create VM and register allocator
    PTXVM vm;
    ASSERT_TRUE(vm.initialize());
    
    RegisterAllocator& allocator = vm.getRegisterAllocator();
    
    // Parameters from state
    uint32_t numPhysicalRegisters = static_cast<uint32_t>(state.range(0));
    uint32_t numWarps = static_cast<uint32_t>(state.range(1));
    uint32_t threadsPerWarp = static_cast<uint32_t>(state.range(2));
    
    // Benchmark loop
    for (auto _ : state) {
        // Allocate registers
        bool result = allocator.allocateRegisters(numPhysicalRegisters, numWarps, threadsPerWarp);
        
        // Check result to prevent compiler optimizations
        benchmark::DoNotOptimize(result);
        
        // Count the number of virtual registers allocated
        state.counters["VirtualRegisters"] = allocator.getTotalVirtualRegisters();
        
        // Count utilization
        state.counters["Utilization"] = allocator.getRegisterUtilization();
    }
    
    // Set items per iteration
    state.SetItemsProcessed(state.iterations());
}

// Register the benchmark
BENCHMARK(BM_RegisterAllocation_Simple)
    ->Args({16, 1, 32})   // Small configuration
    ->Args({32, 4, 32})   // Medium configuration
    ->Args({64, 8, 32})   // Larger configuration
    ->Args({128, 16, 32}) // Very large configuration
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime()
    ->Iterations(100);

// Benchmark register mapping performance
static void BM_RegisterMapping_Performance(benchmark::State& state) {
    // Setup - create VM and register allocator
    PTXVM vm;
    ASSERT_TRUE(vm.initialize());
    
    RegisterAllocator& allocator = vm.getRegisterAllocator();
    
    // Initial allocation - large enough to require spilling
    allocator.allocateRegisters(16, 4, 32);
    
    // Benchmark loop
    for (auto _ : state) {
        // Map a range of virtual registers
        for (uint32_t i = 0; i < 1024; ++i) {
            // Alternate thread IDs to simulate real usage
            uint32_t threadId = i % 32;
            
            // Map virtual register to physical
            RegisterID physicalReg = allocator.mapVirtualToPhysical(i, threadId);
            
            // Check result to prevent compiler optimizations
            benchmark::DoNotOptimize(physicalReg);
        }
    }
    
    // Set items per iteration
    state.SetItemsProcessed(state.iterations() * 1024);
}

// Register the benchmark
BENCHMARK(BM_RegisterMapping_Performance)->Unit(benchmark::kMicrosecond);

// Benchmark register allocation with spilling
static void BM_RegisterAllocation_WithSpill(benchmark::State& state) {
    // Setup - create VM and register allocator
    PTXVM vm;
    ASSERT_TRUE(vm.initialize());
    
    RegisterAllocator& allocator = vm.getRegisterAllocator();
    
    // Parameters from state
    uint32_t numPhysicalRegisters = static_cast<uint32_t>(state.range(0));
    uint32_t numWarps = static_cast<uint32_t>(state.range(1));
    uint32_t threadsPerWarp = static_cast<uint32_t>(state.range(2));
    
    // Calculate virtual registers needed
    uint32_t totalVirtualRegisters = numPhysicalRegisters * 2;  // Force spilling
    
    // Benchmark loop
    for (auto _ : state) {
        // Allocate registers
        bool result = allocator.allocateRegisters(numPhysicalRegisters, numWarps, threadsPerWarp);
        
        // Check result to prevent compiler optimizations
        benchmark::DoNotOptimize(result);
        
        // Count the number of virtual registers allocated
        state.counters["VirtualRegisters"] = allocator.getTotalVirtualRegisters();
        
        // Count spill operations
        state.counters["SpillOperations"] = vm.getPerformanceCounters().getCount(PerformanceCounters::SPILL_OPERATIONS);
    }
    
    // Set items per iteration
    state.SetItemsProcessed(state.iterations());
}

// Register the benchmark
BENCHMARK(BM_RegisterAllocation_WithSpill)
    ->Args({8, 1, 32})    // Low physical registers
    ->Args({16, 2, 32})   // Moderate configuration
    ->Args({32, 4, 32})   // Larger configuration
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime()
    ->Iterations(50);

// Benchmark register utilization
static void BM_RegisterUtilization_Measurement(benchmark::State& state) {
    // Setup - create VM and register allocator
    PTXVM vm;
    ASSERT_TRUE(vm.initialize());
    
    RegisterAllocator& allocator = vm.getRegisterAllocator();
    
    // Allocate a standard configuration
    allocator.allocateRegisters(32, 4, 32);
    
    // Simulate register usage by allocating and freeing
    std::vector<RegisterID> allocatedRegs;
    
    // Benchmark loop
    for (auto _ : state) {
        // In each iteration, allocate some registers and measure utilization
        
        // Simple pattern: allocate 4 registers, then free them
        // This creates varying utilization patterns
        for (int i = 0; i < 100; ++i) {
            // Allocate some registers
            for (uint32_t reg = 0; reg < 8; ++reg) {
                RegisterID physicalReg = allocator.mapVirtualToPhysical(reg, 0);
                if (physicalReg != INVALID_REGISTER) {
                    allocatedRegs.push_back(physicalReg);
                }
            }
            
            // Measure utilization
            double utilization = allocator.getRegisterUtilization();
            
            // Record utilization
            state.counters["Utilization"] += utilization;
            
            // Free half of the registers
            allocatedRegs.resize(allocatedRegs.size() / 2);
        }
    }
    
    // Average utilization across iterations
    state.counters["Utilization"] /= state.iterations();
    
    // Set items per iteration
    state.SetItemsProcessed(state.iterations() * 100);
}

// Register the benchmark
BENCHMARK(BM_RegisterUtilization_Measurement)->Unit(benchmark::kMicrosecond);

// Benchmark context switching overhead
static void BM_ContextSwitching_Overhead(benchmark::State& state) {
    // Setup - create VM and register allocator
    PTXVM vm;
    ASSERT_TRUE(vm.initialize());
    
    RegisterAllocator& allocator = vm.getRegisterAllocator();
    
    // Allocate registers
    allocator.allocateRegisters(32, 4, 32);
    
    // Fill registers with data
    for (uint32_t warpId = 0; warpId < 4; ++warpId) {
        for (uint32_t threadId = 0; threadId < 32; ++threadId) {
            // Save register state
            bool result = allocator.saveRegisterState(warpId, threadId);
            
            // Check result to prevent compiler optimizations
            benchmark::DoNotOptimize(result);
        }
    }
    
    // Benchmark loop
    for (auto _ : state) {
        // Simulate context switches between warps and threads
        for (uint32_t warpId = 0; warpId < 4; ++warpId) {
            for (uint32_t threadId = 0; threadId < 32; ++threadId) {
                // Save register state
                bool saveResult = allocator.saveRegisterState(warpId, threadId);
                
                // Restore register state
                bool restoreResult = allocator.restoreRegisterState(warpId, threadId);
                
                // Check results to prevent compiler optimizations
                benchmark::DoNotOptimize(saveResult);
                benchmark::DoNotOptimize(restoreResult);
                
                // Increment counters
                if (saveResult) state.counters["Saves"]++;
                if (restoreResult) state.counters["Restores"]++;
            }
        }
    }
    
    // Set items per iteration
    state.SetItemsProcessed(state.iterations() * 4 * 32);
}

// Register the benchmark
BENCHMARK(BM_ContextSwitching_Overhead)->Unit(benchmark::kMicrosecond);

// Benchmark instruction execution with different register pressures
static void BM_InstructionExecution_RegisterPressure(benchmark::State& state) {
    // Setup - create VM and register allocator
    PTXVM vm;
    ASSERT_TRUE(vm.initialize());
    
    RegisterAllocator& allocator = vm.getRegisterAllocator();
    
    // Get reference to executor
    PTXExecutor& executor = vm.getExecutor();
    
    // Parameters from state
    uint32_t numPhysicalRegisters = static_cast<uint32_t>(state.range(0));
    uint32_t numWarps = static_cast<uint32_t>(state.range(1));
    uint32_t threadsPerWarp = static_cast<uint32_t>(state.range(2));
    
    // Create a simple ADD instruction
    DecodedInstruction instr;
    instr.type = InstructionTypes::ADD;
    
    // Destination register
    instr.dest.type = OperandType::REGISTER;
    instr.dest.registerIndex = 0;
    
    // Source registers
    Operand src1;
    src1.type = OperandType::REGISTER;
    src1.registerIndex = 1;
    
    Operand src2;
    src2.type = OperandType::REGISTER;
    src2.registerIndex = 2;
    
    instr.sources.push_back(src1);
    instr.sources.push_back(src2);
    
    // Initialize register bank
    RegisterBank& registerBank = vm.getRegisterBank();
    registerBank.writeRegister(1, 42);
    registerBank.writeRegister(2, 24);
    
    // Allocate registers
    allocator.allocateRegisters(numPhysicalRegisters, numWarps, threadsPerWarp);
    
    // Benchmark loop
    for (auto _ : state) {
        // Execute the instruction
        bool result = vm.executeDecodedInstruction(instr);
        
        // Check result to prevent compiler optimizations
        benchmark::DoNotOptimize(result);
        
        // Count execution
        if (result) {
            state.counters["Instructions"]++;
        }
    }
    
    // Set items per iteration
    state.SetItemsProcessed(state.iterations());
}

// Register the benchmark
BENCHMARK(BM_InstructionExecution_RegisterPressure)
    ->Args({8, 1, 32})    // High pressure, limited registers
    ->Args({16, 1, 32})   // Moderate pressure
    ->Args({32, 4, 32})   // Lower pressure with more warps
    ->Unit(benchmark::kMicrosecond);

// Benchmark scheduler with different scheduling algorithms
static void BM_Scheduler_Algorithms(benchmark::State& state) {
    // Setup - create VM and instruction scheduler
    PTXVM vm;
    ASSERT_TRUE(vm.initialize());
    
    InstructionScheduler scheduler;
    
    // Create a sample set of instructions
    std::vector<DecodedInstruction> instructions;
    
    // Generate a mix of instructions
    for (int i = 0; i < 100; ++i) {
        DecodedInstruction instr;
        
        // Alternate instruction types to create dependencies
        if (i % 5 == 0) {
            // Branch instruction
            instr.type = InstructionTypes::BRA;
            
            // For simplicity, we don't fully implement branches here
        } else if (i % 3 == 0) {
            // Memory load
            instr.type = InstructionTypes::LD;
            
            // Destination register
            instr.dest.type = OperandType::REGISTER;
            instr.dest.registerIndex = i % 16;
            
            // Source is memory
            Operand memOp;
            memOp.type = OperandType::MEMORY;
            memOp.address = i * 4;
            instr.sources.push_back(memOp);
        } else {
            // Math instruction
            instr.type = InstructionTypes::ADD;
            
            // Destination register
            instr.dest.type = OperandType::REGISTER;
            instr.dest.registerIndex = i % 16;
            
            // Source registers
            Operand src1;
            src1.type = OperandType::REGISTER;
            src1.registerIndex = (i + 1) % 16;
            
            Operand src2;
            src2.type = OperandType::REGISTER;
            src2.registerIndex = (i + 2) % 16;
            
            instr.sources.push_back(src1);
            instr.sources.push_back(src2);
        }
        
        instructions.push_back(instr);
    }
    
    // Parameters from state
    SchedulingAlgorithm algorithm = static_cast<SchedulingAlgorithm>(state.range(0));
    
    // Benchmark loop
    for (auto _ : state) {
        // Set scheduling algorithm
        scheduler.setSchedulingAlgorithm(algorithm);
        
        // Schedule instructions
        std::vector<ScheduledInstruction> scheduled;
        bool result = scheduler.scheduleInstructions(instructions, scheduled, 4, 32);
        
        // Check result to prevent compiler optimizations
        benchmark::DoNotOptimize(result);
        
        // Count cycles used
        if (result && !scheduled.empty()) {
            size_t firstCycle = scheduled.front().scheduledCycle;
            size_t lastCycle = scheduled.back().scheduledCycle;
            state.counters["Cycles"] = lastCycle - firstCycle + 1;
            
            // Calculate IPC
            double ipc = static_cast<double>(instructions.size()) / (lastCycle - firstCycle + 1);
            state.counters["IPC"] = ipc;
        }
    }
    
    // Set items per iteration
    state.SetItemsProcessed(state.iterations());
}

// Register the benchmark
BENCHMARK(BM_Scheduler_Algorithms)->Arg(SCHEDULING_SIMPLE_INORDER)->Arg(SCHEDULING_LIST_BASED)->Unit(benchmark::kMicrosecond);

// Benchmark memory operations with different register allocations
static void BM_MemoryOperations_RegisterAllocation(benchmark::State& state) {
    // Setup - create VM and register allocator
    PTXVM vm;
    ASSERT_TRUE(vm.initialize());
    
    RegisterAllocator& allocator = vm.getRegisterAllocator();
    
    // Parameters from state
    uint32_t numPhysicalRegisters = static_cast<uint32_t>(state.range(0));
    uint32_t numWarps = static_cast<uint32_t>(state.range(1));
    uint32_t threadsPerWarp = static_cast<uint32_t>(state.range(2));
    
    // Allocate registers
    allocator.allocateRegisters(numPhysicalRegisters, numWarps, threadsPerWarp);
    
    // Get reference to memory subsystem
    MemorySubsystem& memory = vm.getMemorySubsystem();
    
    // Allocate some memory
    const size_t MEM_SIZE = 1024;
    void* buffer = memory.allocateMemory(MemorySpace::GLOBAL, MEM_SIZE);
    
    // Fill memory with known pattern
    if (buffer) {
        memset(buffer, 0xAA, MEM_SIZE);
    }
    
    // Benchmark loop
    for (auto _ : state) {
        // Perform memory operations
        for (int i = 0; i < 100; ++i) {
            // Read from memory
            uint64_t value = memory.read<uint64_t>(MemorySpace::GLOBAL, reinterpret_cast<uint64_t>(buffer) + (i % (MEM_SIZE / sizeof(uint64_t)) * sizeof(uint64_t));
            
            // Write back modified value
            memory.write<uint64_t>(MemorySpace::GLOBAL, reinterpret_cast<uint64_t>(buffer) + (i % (MEM_SIZE / sizeof(uint64_t)) * sizeof(uint64_t)), value + 1);
            
            // Prevent compiler optimizations
            benchmark::DoNotOptimize(value);
            benchmark::ClobberMemory();
        }
        
        // Count operations
        state.counters["MemoryOps"] = 100;
        
        // Count spillover operations
        state.counters["Spills"] = vm.getPerformanceCounters().getCount(PerformanceCounters::SPILL_OPERATIONS);
    }
    
    // Set items per iteration
    state.SetItemsProcessed(state.iterations() * 100);
}

// Register the benchmark
BENCHMARK(BM_MemoryOperations_RegisterAllocation)
    ->Args({8, 1, 32})     // High spillover
    ->Args({16, 1, 32})    // Moderate spillover
    ->Args({32, 4, 32})    // Minimal spillover
    ->Unit(benchmark::kMicrosecond);

// Run all benchmarks when this file is linked
BENCHMARK_MAIN();