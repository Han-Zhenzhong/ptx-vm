#include "instruction_scheduler.hpp"
#include <iostream>
#include <stdexcept>
#include <unordered_map>
#include <queue>

// Private implementation class
class InstructionScheduler::Impl {
public:
    Impl() {
        // Initialize default instruction latencies
        initializeDefaultLatencies();
    }
    
    ~Impl() = default;
    
    // Set the scheduling algorithm to use
    void setSchedulingAlgorithm(SchedulingAlgorithm algorithm) {
        m_schedulingAlgorithm = algorithm;
    }
    
    // Schedule instructions for execution
    bool scheduleInstructions(const std::vector<DecodedInstruction>& instructions,
                            std::vector<ScheduledInstruction>& scheduledInstructions,
                            uint32_t numWarps,
                            uint32_t threadsPerWarp) {
        // Clear output
        scheduledInstructions.clear();
        
        // Store parameters
        m_numWarps = numWarps;
        m_threadsPerWarp = threadsPerWarp;
        
        // Assign warps and threads to instructions
        assignWarpThreadIds(instructions);
        
        // Based on the selected algorithm, perform instruction scheduling
        switch (m_schedulingAlgorithm) {
            case SCHEDULING_SIMPLE_INORDER:
                return simpleInOrderScheduling(instructions, scheduledInstructions);
                
            case SCHEDULING_LIST_BASED:
                return listBasedScheduling(instructions, scheduledInstructions);
                
            case SCHEDULING_WARP_SPECIALIZED:
                return warpSpecializedScheduling(instructions, scheduledInstructions);
                
            case SCHEDULING_DYNAMIC_REGISTER_ALLOC:
                return dynamicRegisterAllocationScheduling(instructions, scheduledInstructions);
                
            default:
                return simpleInOrderScheduling(instructions, scheduledInstructions);
        }
    }
    
    // Get instruction latency information
    size_t getInstructionLatency(InstructionType type) const {
        auto it = m_instructionLatencies.find(type);
        if (it != m_instructionLatencies.end()) {
            return it->second;
        }
        return 1;  // Default latency of 1 cycle
    }
    
    // Set instruction latency information
    void setInstructionLatency(InstructionType type, size_t cycles) {
        m_instructionLatencies[type] = cycles;
    }
    
    // Get register usage information for an instruction
    void getRegisterUsage(const DecodedInstruction& instruction,
                         std::vector<RegisterID>& inputRegisters,
                         std::vector<RegisterID>& outputRegisters) const {
        // Clear output vectors
        inputRegisters.clear();
        outputRegisters.clear();
        
        // Get output register (destination)
        if (instruction.dest.type == OperandType::REGISTER) {
            outputRegisters.push_back(instruction.dest.registerIndex);
        }
        
        // Get input registers (sources)
        for (const auto& source : instruction.sources) {
            if (source.type == OperandType::REGISTER) {
                inputRegisters.push_back(source.registerIndex);
            }
        }
    }
    
    // Get last scheduling statistics
    const std::unordered_map<std::string, double>& getSchedulingStats() const {
        return m_schedulingStats;
    }
    
private:
    // Core configuration
    SchedulingAlgorithm m_schedulingAlgorithm = SCHEDULING_SIMPLE_INORDER;
    uint32_t m_numWarps = 0;
    uint32_t m_threadsPerWarp = 0;
    
    // Instruction latency table
    std::unordered_map<InstructionType, size_t> m_instructionLatencies;
    
    // Scheduling statistics
    std::unordered_map<std::string, double> m_schedulingStats;
    
    // Initialize default instruction latencies
    void initializeDefaultLatencies() {
        // Basic arithmetic operations
        m_instructionLatencies[InstructionTypes::ADD] = 1;
        m_instructionLatencies[InstructionTypes::SUB] = 1;
        m_instructionLatencies[InstructionTypes::MUL] = 4;  // Multiply takes more cycles
        m_instructionLatencies[InstructionTypes::DIV] = 8;  // Divide is expensive
        
        // Memory operations
        m_instructionLatencies[InstructionTypes::LD] = 2;   // Load from memory
        m_instructionLatencies[InstructionTypes::ST] = 2;   // Store to memory
        
        // Control flow
        m_instructionLatencies[InstructionTypes::BRA] = 1;  // Branch instruction
        m_instructionLatencies[InstructionTypes::EXIT] = 1;  // Exit instruction
        
        // Other operations
        m_instructionLatencies[InstructionTypes::MOV] = 1;  // Move operation
        m_instructionLatencies[InstructionTypes::NOP] = 1;    // No-op
    }
    
    // Assign warp/thread IDs to instructions based on warp count and threads per warp
    void assignWarpThreadIds(const std::vector<DecodedInstruction>& instructions) {
        // For now, just assign all instructions to warp 0, thread 0
        // In real implementation, this would be more complex
        m_instructionWarpMap.clear();
        m_instructionWarpMap.resize(instructions.size(), 0);
        
        m_instructionThreadMap.clear();
        m_instructionThreadMap.resize(instructions.size(), 0);
    }
    
    // Simple in-order scheduling
    bool simpleInOrderScheduling(const std::vector<DecodedInstruction>& instructions,
                               std::vector<ScheduledInstruction>& scheduledInstructions) {
        // For now, just schedule instructions in original order with no optimization
        // Each instruction starts one cycle after the previous
        
        scheduledInstructions.resize(instructions.size());
        
        for (size_t i = 0; i < instructions.size(); ++i) {
            ScheduledInstruction& schedInstr = scheduledInstructions[i];
            schedInstr.originalIndex = i;
            schedInstr.scheduledCycle = i;
            schedInstr.instruction = instructions[i];
            schedInstr.warpId = m_instructionWarpMap[i];
            schedInstr.threadId = m_instructionThreadMap[i];
            
            // No dependencies yet
            schedInstr.dependencies.clear();
        }
        
        // Update scheduling statistics
        m_schedulingStats["scheduling_cycles"] = instructions.size();
        m_schedulingStats["instructions_per_cycle"] = 1.0;
        
        return true;
    }
    
    // List-based scheduling with dependency tracking
    bool listBasedScheduling(const std::vector<DecodedInstruction>& instructions,
                           std::vector<ScheduledInstruction>& scheduledInstructions) {
        // This will implement a basic list-based scheduling algorithm that considers dependencies
        
        // Build dependency graph
        std::vector<std::vector<size_t>> dependencies;
        buildDependencyGraph(instructions, dependencies);
        
        // Calculate earliest available cycle for each instruction
        std::vector<size_t> earliestCycle(instructions.size(), 0);
        
        // Calculate depth of each node in graph (used to prioritize instructions)
        std::vector<size_t> depths(instructions.size(), 0);
        calculateDepths(dependencies, depths);
        
        // Schedule instructions
        std::vector<bool> scheduled(instructions.size(), false);
        scheduledInstructions.clear();
        
        // Current cycle
        size_t currentCycle = 0;
        
        // Instructions ready to be scheduled
        std::priority_queue<std::pair<size_t, size_t>, 
                          std::vector<std::pair<size_t, size_t>>, 
                          decltype([](const std::pair<size_t, size_t>& a, 
                                    const std::pair<size_t, size_t>& b) {
            // Prioritize by depth (higher depth first)
            return a.second < b.second;
        }) readyQueue;
        
        // Initialize ready queue with instructions that have no dependencies
        for (size_t i = 0; i < instructions.size(); ++i) {
            if (dependencies[i].empty()) {
                readyQueue.emplace(i, depths[i]);
            }
        }
        
        // Process instructions in priority order
        while (!readyQueue.empty()) {
            // Get instruction with highest depth
            size_t instrIndex = readyQueue.top().first;
            readyQueue.pop();
            
            // Schedule instruction at current cycle
            ScheduledInstruction schedInstr;
            schedInstr.originalIndex = instrIndex;
            schedInstr.scheduledCycle = currentCycle;
            schedInstr.instruction = instructions[instrIndex];
            schedInstr.warpId = m_instructionWarpMap[instrIndex];
            schedInstr.threadId = m_instructionThreadMap[instrIndex];
            
            // Record dependencies
            for (const auto& dep : dependencies[instrIndex]) {
                InstructionDependency dependency;
                dependency.sourceIndex = dep;
                dependency.registerID = 0;  // Simplified
                
                // Determine dependency type
                // In real implementation, we would analyze register usage
                dependency.type = InstructionDependency::RAW;
                
                schedInstr.dependencies.push_back(dependency);
            }
            
            scheduledInstructions.push_back(schedInstr);
            scheduled[instrIndex] = true;
            
            // Add dependent instructions to ready queue
            for (size_t i = 0; i < instructions.size(); ++i) {
                if (!scheduled[i]) {
                    // Check if all dependencies are satisfied
                    bool canSchedule = true;
                    for (const auto& dep : dependencies[i]) {
                        if (!scheduled[dep]) {
                            canSchedule = false;
                            break;
                        }
                    }
                    
                    if (canSchedule) {
                        // Calculate depth for this instruction
                        size_t depth = 0;
                        for (const auto& dep : dependencies[i]) {
                            for (const auto& si : scheduledInstructions) {
                                if (si.originalIndex == dep) {
                                    depth = std::max(depth, si.scheduledCycle + 1);
                                }
                            }
                        }
                        
                        // Add to ready queue
                        readyQueue.emplace(i, depth);
                    }
                }
            }
            
            currentCycle++;
        }
        
        // Sort scheduled instructions by scheduled cycle
        std::sort(scheduledInstructions.begin(), scheduledInstructions.end(), 
                 [](const ScheduledInstruction& a, const ScheduledInstruction& b) {
            return a.scheduledCycle < b.scheduledCycle;
        });
        
        // Update scheduling statistics
        m_schedulingStats["scheduling_cycles"] = currentCycle;
        m_schedulingStats["instructions_per_cycle"] = static_cast<double>(instructions.size()) / currentCycle;
        
        return !scheduledInstructions.empty();
    }
    
    // Warp-specialized scheduling
    bool warpSpecializedScheduling(const std::vector<DecodedInstruction>& instructions,
                                 std::vector<ScheduledInstruction>& scheduledInstructions) {
        // This will implement a scheduling algorithm optimized for warp execution
        
        // First group instructions by warp
        std::unordered_map<uint32_t, std::vector<size_t>> warpInstructions;
        
        for (size_t i = 0; i < instructions.size(); ++i) {
            uint32_t warpId = m_instructionWarpMap[i];
            warpInstructions[warpId].push_back(i);
        }
        
        // Schedule each warp's instructions
        scheduledInstructions.clear();
        size_t currentCycle = 0;
        
        // Keep track of when each register will be available
        std::unordered_map<RegisterID, size_t> registerAvailability;
        
        // For each warp, schedule its instructions
        for (auto& warpEntry : warpInstructions) {
            const std::vector<size_t>& warpInstrIndices = warpEntry.second;
            
            // Schedule instructions for this warp
            for (size_t idx : warpInstrIndices) {
                const DecodedInstruction& instr = instructions[idx];
                
                // Find when all input registers will be available
                std::vector<RegisterID> inputRegisters;
                std::vector<RegisterID> outputRegisters;
                getRegisterUsage(instr, inputRegisters, outputRegisters);
                
                size_t earliestCycle = currentCycle;
                
                // Wait for input registers
                for (RegisterID reg : inputRegisters) {
                    if (registerAvailability.find(reg) != registerAvailability.end()) {
                        earliestCycle = std::max(earliestCycle, registerAvailability[reg]);
                    }
                }
                
                // Schedule instruction
                ScheduledInstruction schedInstr;
                schedInstr.originalIndex = idx;
                schedInstr.scheduledCycle = earliestCycle;
                schedInstr.instruction = instr;
                schedInstr.warpId = m_instructionWarpMap[idx];
                schedInstr.threadId = m_instructionThreadMap[idx];
                
                // Determine dependencies
                // In real implementation, this would be more detailed
                schedInstr.dependencies.clear();
                
                scheduledInstructions.push_back(schedInstr);
                
                // Update register availability
                size_t latency = getInstructionLatency(instr.type);
                size_t finishCycle = earliestCycle + latency;
                
                for (RegisterID reg : outputRegisters) {
                    registerAvailability[reg] = finishCycle;
                }
                
                // Output registers are not available until finishCycle
                currentCycle = std::max(currentCycle, finishCycle);
            }
            
            // Increment cycle for warp boundary
            currentCycle++;
        }
        
        // Sort by scheduled cycle
        std::sort(scheduledInstructions.begin(), scheduledInstructions.end(), 
                 [](const ScheduledInstruction& a, const ScheduledInstruction& b) {
            return a.scheduledCycle < b.scheduledCycle;
        });
        
        // Update scheduling statistics
        m_schedulingStats["scheduling_cycles"] = currentCycle;
        m_schedulingStats["instructions_per_cycle"] = static_cast<double>(instructions.size()) / currentCycle;
        
        return !scheduledInstructions.empty();
    }
    
    // Dynamic register allocation scheduling
    bool dynamicRegisterAllocationScheduling(const std::vector<DecodedInstruction>& instructions,
                                            std::vector<ScheduledInstruction>& scheduledInstructions) {
        // This implements a scheduling approach that considers register allocation
        
        // We'll use a simple list-based approach but consider register pressure
        
        // Build dependency graph
        std::vector<std::vector<size_t>> dependencies;
        buildDependencyGraph(instructions, dependencies);
        
        // Calculate earliest available cycle for each instruction
        std::vector<size_t> earliestCycle(instructions.size(), 0);
        
        // Calculate depth of each node in graph
        std::vector<size_t> depths(instructions.size(), 0);
        calculateDepths(dependencies, depths);
        
        // Schedule instructions
        std::vector<bool> scheduled(instructions.size(), false);
        scheduledInstructions.clear();
        
        // Current cycle
        size_t currentCycle = 0;
        
        // Keep track of when each register will be available
        std::unordered_map<RegisterID, size_t> registerAvailability;
        
        // Register usage tracking
        std::unordered_map<RegisterID, size_t> registerUseCount;
        
        // Instructions ready to be scheduled
        std::priority_queue<std::pair<size_t, size_t>, 
                          std::vector<std::pair<size_t, size_t>>, 
                          decltype([](const std::pair<size_t, size_t>& a, 
                                    const std::pair<size_t, size_t>& b) {
            // Prioritize by depth (higher depth first), then by register pressure reduction
            if (a.second != b.second) {
                return a.second < b.second;
            }
                
            // If depth is equal, prioritize instructions using high-pressure registers
            size_t aPressure = 0;
            size_t bPressure = 0;
            
            const DecodedInstruction& instrA = instructions[a.first];
            const DecodedInstruction& instrB = instructions[b.first];
            
            // Count register uses for A
            if (instrA.dest.type == OperandType::REGISTER) {
                aPressure += registerUseCount[instrA.dest.registerIndex];
            }
            for (const auto& src : instrA.sources) {
                if (src.type == OperandType::REGISTER) {
                    aPressure += registerUseCount[src.registerIndex];
                }
            }
            
            // Count register uses for B
            if (instrB.dest.type == OperandType::REGISTER) {
                bPressure += registerUseCount[instrB.registers];
            }
            for (const auto& src : instrB.sources) {
                if (src.type == OperandType::REGISTER) {
                    bPressure += registerUseCount[src.registerIndex];
                }
            }
            
            return aPressure < bPressure;
        }) readyQueue;
        
        // Initialize register use counts
        calculateRegisterUseCounts(instructions, registerUseCount);
        
        // Initialize ready queue with instructions that have no dependencies
        for (size_t i = 0; i < instructions.size(); ++i) {
            if (dependencies[i].empty()) {
                readyQueue.emplace(i, depths[i]);
            }
        }
        
        // Process instructions in priority order
        while (!readyQueue.empty()) {
            // Get instruction with highest priority
            size_t instrIndex = readyQueue.top().first;
            readyQueue.pop();
            
            // Schedule instruction
            ScheduledInstruction schedInstr;
            schedInstr.originalIndex = instrIndex;
            schedInstr.scheduledCycle = currentCycle;
            schedInstr.instruction = instructions[instrIndex];
            schedInstr.warpId = m_instructionWarpMap[instrIndex];
            schedInstr.threadId = m_instructionThreadMap[instrIndex];
            
            // Determine dependencies
            // In real implementation, this would be more detailed
            schedInstr.dependencies.clear();
            
            scheduledInstructions.push_back(schedInstr);
            scheduled[instrIndex] = true;
            
            // Update register availability
            std::vector<RegisterID> inputRegisters;
            std::vector<RegisterID> outputRegisters;
            getRegisterUsage(instructions[instrIndex], inputRegisters, outputRegisters);
            
            size_t latency = getInstructionLatency(instructions[instrIndex].type);
            size_t finishCycle = currentCycle + latency;
            
            for (RegisterID reg : outputRegisters) {
                registerAvailability[reg] = finishCycle;
            }
            
            // Update current cycle
            currentCycle++;
            
            // Add dependent instructions to ready queue
            for (size_t i = 0; i < instructions.size(); ++i) {
                if (!scheduled[i]) {
                    // Check if all dependencies are satisfied
                    bool canSchedule = true;
                    for (const auto& dep : dependencies[i]) {
                        if (!scheduled[dep]) {
                            canSchedule = false;
                            break;
                        }
                    }
                    
                    if (canSchedule && std::find_if(
                        scheduledInstructions.begin(), 
                        scheduledInstructions.end(), 
                        [i](const ScheduledInstruction& si) { return si.originalIndex == i; }) == scheduledInstructions.end()) {
                        // Calculate depth for this instruction
                        size_t depth = 0;
                        for (const auto& dep : dependencies[i]) {
                            for (const auto& si : scheduledInstructions) {
                                if (si.originalIndex == dep) {
                                    depth = std::max(depth, si.scheduledCycle + 1);
                                    break;
                                }
                            }
                        }
                        
                        // Add to ready queue
                        readyQueue.emplace(i, depth);
                    }
                }
            }
        }
        
        // Update scheduling statistics
        m_schedulingStats["scheduling_cycles"] = currentCycle;
        m_schedulingStats["instructions_per_cycle"] = static_cast<double>(instructions.size()) / currentCycle;
        
        return !scheduledInstructions.empty();
    }
    
    // Build dependency graph for instructions
    void buildDependencyGraph(const std::vector<DecodedInstruction>& instructions,
                            std::vector<std::vector<size_t>>& dependencies) {
        // Build a simple data dependency graph
        dependencies.clear();
        dependencies.resize(instructions.size());
        
        // Track register definitions
        std::unordered_map<RegisterID, size_t> registerDefinitions;
        
        for (size_t i = 0; i < instructions.size(); ++i) {
            const DecodedInstruction& instr = instructions[i];
            
            // Check for data dependencies
            std::vector<RegisterID> inputRegisters;
            std::vector<RegisterID> outputRegisters;
            getRegisterUsage(instr, inputRegisters, outputRegisters);
            
            // RAW (Read After Write) dependencies
            for (RegisterID reg : inputRegisters) {
                if (registerDefinitions.find(reg) != registerDefinitions.end()) {
                    size_t dep = registerDefinitions[reg];
                    dependencies[i].push_back(dep);
                }
            }
            
            // WAW (Write After Write) dependencies
            for (RegisterID reg : outputRegisters) {
                if (registerDefinitions.find(reg) != registerDefinitions.end()) {
                    size_t dep = registerDefinitions[reg];
                    dependencies[i].push_back(dep);
                }
            }
            
            // Update register definitions
            for (RegisterID reg : outputRegisters) {
                registerDefinitions[reg] = i;
            }
        }
        
        // Remove duplicates from dependencies
        for (auto& deps : dependencies) {
            std::sort(deps.begin(), deps.end());
            deps.erase(std::unique(deps.begin(), deps.end()), deps.end());
        }
    }
    
    // Calculate depth of each node in dependency graph
    void calculateDepths(const std::vector<std::vector<size_t>>& dependencies, 
                       std::vector<size_t>& depths) {
        // Depth is the length of the longest path to this node
        for (size_t i = 0; i < dependencies.size(); ++i) {
            size_t maxDepth = 0;
            
            for (size_t dep : dependencies[i]) {
                maxDepth = std::max(maxDepth, depths[dep] + 1);
            }
            
            depths[i] = maxDepth;
        }
    }
    
    // Calculate register use counts
    void calculateRegisterUseCounts(const std::vector<DecodedInstruction>& instructions,
                                 std::unordered_map<RegisterID, size_t>& useCounts) {
        // Count how many times each register is used
        useCounts.clear();
        
        for (const auto& instr : instructions) {
            std::vector<RegisterID> inputRegisters;
            std::vector<RegisterID> outputRegisters;
            getRegisterUsage(instr, inputRegisters, outputRegisters);
            
            // Count inputs
            for (RegisterID reg : inputRegisters) {
                useCounts[reg]++;
            }
            
            // Count outputs
            for (RegisterID reg : outputRegisters) {
                useCounts[reg]++;
            }
        }
    }
    
    // Get register usage for an instruction
    void getRegisterUsage(const DecodedInstruction& instruction,
                         std::vector<RegisterID>& inputRegisters,
                         std::vector<RegisterID>& outputRegisters) {
        // Clear output vectors
        inputRegisters.clear();
        outputRegisters.clear();
        
        // Get output register (destination)
        if (instruction.dest.type == OperandType::REGISTER) {
            outputRegisters.push_back(instruction.dest.registerIndex);
        }
        
        // Get input registers (sources)
        for (const auto& source : instruction.sources) {
            if (source.type == OperandType::REGISTER) {
                inputRegisters.push_back(source.registerIndex);
            }
        }
    }
};

InstructionScheduler::InstructionScheduler() : pImpl(std::make_unique<Impl>()) {}

InstructionScheduler::~InstructionScheduler() = default;

void InstructionScheduler::setSchedulingAlgorithm(SchedulingAlgorithm algorithm) {
    pImpl->setSchedulingAlgorithm(algorithm);
}

bool InstructionScheduler::scheduleInstructions(const std::vector<DecodedInstruction>& instructions,
                                          std::vector<ScheduledInstruction>& scheduledInstructions,
                                          uint32_t numWarps,
                                          uint32_t threadsPerWarp) {
    return pImpl->scheduleInstructions(instructions, scheduledInstructions, numWarps, threadsPerWarp);
}

size_t InstructionScheduler::getInstructionLatency(InstructionType type) const {
    return pImpl->getInstructionLatency(type);
}

void InstructionScheduler::setInstructionLatency(InstructionType type, size_t cycles) {
    pImpl->setInstructionLatency(type, cycles);
}

void InstructionScheduler::getRegisterUsage(const DecodedInstruction& instruction,
                                        std::vector<RegisterID>& inputRegisters,
                                        std::vector<RegisterID>& outputRegisters) {
    pImpl->getRegisterUsage(instruction, inputRegisters, outputRegisters);
}

const std::unordered_map<std::string, double>& InstructionScheduler::getSchedulingStats() const {
    return pImpl->getSchedulingStats();
}