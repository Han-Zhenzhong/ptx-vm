#include <gtest/gtest.h>
#include <benchmark/benchmark.h>
#include "vm.hpp"
#include "examples/simple_math_example.ptx"
#include "examples/control_flow_example.ptx"
#include "examples/memory_ops_example.ptx"

// Test fixture for performance tests
class PerformanceTest : public ::testing::Test {
protected:
    PTXVM vm;
    
    void SetUp() override {
        ASSERT_TRUE(vm.initialize());
    }
    
    void TearDown() override {
        // Clean up if needed
    }
};

// Benchmark different execution scenarios
static void BM_Execution_SimpleMath(benchmark::State& state) {
    PTXVM vm;
    ASSERT_TRUE(vm.initialize());
    
    // Load and execute the simple math example
    for (auto _ : state) {
        bool result = vm.loadAndExecuteProgram("examples/simple_math_example.ptx");
        
        // Check result to prevent compiler optimizations
        benchmark::DoNotOptimize(result);
        
        // Count instructions executed
        state.counters["Instructions"] = vm.getPerformanceCounters().getCount(PerformanceCounters::INSTRUCTIONS_EXECUTED);
        
        // Count cycles
        state.counters["Cycles"] = vm.getPerformanceCounters().getCount(PerformanceCounters::CYCLES);
    }
    
    // Set items per iteration
    state.SetItemsProcessed(state.iterations());
}

// Register the benchmark
BENCHMARK(BM_Execution_SimpleMath)->Unit(benchmark::kMillisecond);

// Benchmark control flow execution
static void BM_Execution_ControlFlow(benchmark::State& state) {
    PTXVM vm;
    ASSERT_TRUE(vm.initialize());
    
    // Load and execute the control flow example
    for (auto _ : state) {
        bool result = vm.loadAndExecuteProgram("examples/control_flow_example.ptx");
        
        // Check result to prevent compiler optimizations
        benchmark::DoNotOptimize(result);
        
        // Count instructions executed
        state.counters["Instructions"] = vm.getPerformanceCounters().getCount(PerformanceCounters::INSTRUCTIONS_EXECUTED);
        
        // Count divergent branches
        state.counters["DivergentBranches"] = vm.getPerformanceCounters().getCount(PerformanceCounters::DIVERGENT_BRANCHES);
        
        // Count warp switches
        state.counters["WarpSwitches"] = vm.getPerformanceCounters().getCount(PerformanceCounters::WARP_SWITCHES);
    }
    
    // Set items per iteration
    state.SetItemsProcessed(state.iterations());
}

// Register the benchmark
BENCHMARK(BM_Execution_ControlFlow)->Unit(benchmark::kMillisecond);

// Benchmark memory operations
static void BM_Execution_MemoryOps(benchmark::State& state) {
    PTXVM vm;
    ASSERT_TRUE(vm.initialize());
    
    // Load and execute the memory operations example
    for (auto _ : state) {
        bool result = vm.loadAndExecuteProgram("examples/memory_ops_example.ptx");
        
        // Check result to prevent compiler optimizations
        benchmark::DoNotOptimize(result);
        
        // Count memory operations
        state.counters["GlobalReads"] = vm.getPerformanceCounters().getCount(PerformanceCounters::GLOBAL_MEMORY_READS);
        state.counters["GlobalWrites"] = vm.getPerformanceCounters().getCount(PerformanceCounters::GLOBAL_MEMORY_WRITES);
        state.counters["SharedReads"] = vm.getPerformanceCounters().getCount(PerformanceCounters::SHARED_MEMORY_READS);
        state.counters["SharedWrites"] = vm.getPerformanceCounters().getCount(PerformanceCounters::SHARED_MEMORY_WRITES);
        
        // Count cache misses
        state.counters["ICacheMisses"] = vm.getPerformanceCounters().getCount(PerformanceCounters::ICACHE_MISSES);
        state.counters["DCacheMisses"] = vm.getPerformanceCounters().getCount(PerformanceCounters::DCACHE_MISSES);
    }
    
    // Set items per iteration
    state.SetItemsProcessed(state.iterations());
}

// Register the benchmark
BENCHMARK(BM_Execution_MemoryOps)->Unit(benchmark::kMillisecond);

// Benchmark register allocation efficiency
static void BM_Execution_RegisterAllocation(benchmark::State& state) {
    PTXVM vm;
    ASSERT_TRUE(vm.initialize());
    
    // Get reference to register allocator
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
BENCHMARK(BM_Execution_RegisterAllocation)
    ->Args({16, 1, 32})   // Small configuration
    ->Args({32, 4, 32})   // Medium configuration
    ->Args({64, 8, 32})   // Larger configuration
    ->Args({128, 16, 32}) // Very large configuration
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime()
    ->Iterations(100);

// Benchmark instruction scheduling efficiency
static void BM_Execution_InstructionScheduling(benchmark::State& state) {
    PTXVM vm;
    ASSERT_TRUE(vm.initialize());
    
    // Get reference to instruction scheduler
    InstructionScheduler& scheduler = vm.getExecutor().getInstructionScheduler();
    
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
BENCHMARK(BM_Execution_InstructionScheduling)->Arg(SCHEDULING_SIMPLE_INORDER)->Arg(SCHEDULING_LIST_BASED)->Unit(benchmark::kMicrosecond);

// Benchmark warp execution efficiency
static void BM_Execution_WarpExecution(benchmark::State& state) {
    PTXVM vm;
    ASSERT_TRUE(vm.initialize());
    
    // Get reference to executor and warp scheduler
    PTXExecutor& executor = vm.getExecutor();
    WarpScheduler& scheduler = executor.getWarpScheduler();
    
    // Initialize warp scheduler with default configuration
    scheduler.initialize(4, 32);  // 4 warps, 32 threads per warp
    
    // Create some dummy work for each warp
    for (uint32_t warpId = 0; warpId < 4; ++warpId) {
        // In real implementation, this would be more complex
        // This is just setting up some basic warp state
        scheduler.setActiveThreads(warpId, 0xFFFFFFFF);  // All threads active
        scheduler.setCurrentPC(warpId, 0);  // Start at PC 0
    }
    
    // Benchmark loop
    for (auto _ : state) {
        // Simulate warp execution by selecting warps round-robin
        for (size_t i = 0; i < 1000; ++i) {
            // Select next warp
            uint32_t selectedWarp = scheduler.selectNextWarp();
            
            // In real implementation, we'd do more complex things here
            // This is just simulating warp switching
            if (selectedWarp != 0) {
                state.counters["WarpSwitches"]++;
            }
            
            // Update warp state
            scheduler.setCurrentPC(selectedWarp, i % 100);
            
            // Simulate thread divergence
            if (i % 10 == 0) {
                // Simulate branch divergence
                uint64_t activeMask = 0xFFFFFFFF;  // All threads active
                uint64_t divergentMask = 0xFFFF0000;  // First half takes branch
                
                // Push divergence point
                scheduler.handleBranchDivergence(i % 100, activeMask, divergentMask);
                
                state.counters["Divergences"]++;
            }
        }
    }
    
    // Set items per iteration
    state.SetItemsProcessed(state.iterations() * 1000);
}

// Register the benchmark
BENCHMARK(BM_Execution_WarpExecution)->Unit(benchmark::kMicrosecond);

// Benchmark predicate handling
static void BM_Execution_PredicateHandling(benchmark::State& state) {
    PTXVM vm;
    ASSERT_TRUE(vm.initialize());
    
    // Get reference to predicate handler
    PredicateHandler& predicateHandler = vm.getExecutor().getPredicateHandler();
    
    // Create a sample decoded instruction with predicate
    DecodedInstruction instr;
    instr.type = InstructionTypes::ADD;
    instr.hasPredicate = true;
    instr.predicateIndex = 0;  // Use predicate register 0
    
    // Set initial predicate state
    predicateHandler.setPredicateState(0, state.range(0) % 2 ? true : false);
    
    // Benchmark loop
    for (auto _ : state) {
        // Evaluate predicate
        bool shouldExecute = predicateHandler.shouldExecute(instr);
        
        // Check result to prevent compiler optimizations
        benchmark::DoNotOptimize(shouldExecute);
        
        // Count executions
        if (shouldExecute) {
            state.counters["Executed"]++;
        } else {
            state.counters["Skipped"]++;
        }
        
        // Toggle predicate value for next iteration
        bool currentValue = predicateHandler.getPredicateState(0)->value;
        predicateHandler.setPredicateState(0, !currentValue);
    }
    
    // Set items per iteration
    state.SetItemsProcessed(state.iterations());
}

// Register the benchmark
BENCHMARK(BM_Execution_PredicateHandling)->Range(0, 1)->Unit(benchmark::kNanosecond);

// Run all benchmarks when this file is linked
BENCHMARK_MAIN();