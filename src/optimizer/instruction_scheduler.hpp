#ifndef INSTRUCTION_SCHEDULER_HPP
#define INSTRUCTION_SCHEDULER_HPP

#include <cstdint>
#include <vector>
#include <memory>
#include "instruction_types.hpp"
#include <unordered_map>

class InstructionScheduler;

typedef uint32_t RegisterID;

typedef struct InstructionDependency {
    // The instruction index that this dependency comes from
    size_t sourceIndex;
    
    // The type of dependency
    enum DependencyType {
        RAW,  // Read-after-write
        WAR,  // Write-after-read
        WAW   // Write-after-write
    } type;
    
    // The register involved in the dependency
    RegisterID registerID;
} InstructionDependency;

typedef struct ScheduledInstruction {
    // Index in original instruction stream
    size_t originalIndex;
    
    // Execution cycle when this instruction can be issued
    size_t scheduledCycle;
    
    // Instruction to schedule
    DecodedInstruction instruction;
    
    // Dependencies for this instruction
    std::vector<InstructionDependency> dependencies;
    
    // Which warp/thread this instruction belongs to
    uint32_t warpId;
    uint32_t threadId;
} ScheduledInstruction;

typedef enum {
    SCHEDULING_SIMPLE_INORDER,          // Simple in-order scheduling
    SCHEDULING_LIST_BASED,              // List-based scheduling with dependencies
    SCHEDULING_WARP_SPECIALIZED,         // Warp-specialized scheduling
    SCHEDULING_DYNAMIC_REGISTER_ALLOC    // Dynamic register allocation scheduling
} SchedulingAlgorithm;

class InstructionScheduler {
public:
    // Constructor/destructor
    InstructionScheduler();
    ~InstructionScheduler();

    // Set the scheduling algorithm to use
    void setSchedulingAlgorithm(SchedulingAlgorithm algorithm);

    // Schedule instructions for execution
    bool scheduleInstructions(const std::vector<DecodedInstruction>& instructions,
                            std::vector<ScheduledInstruction>& scheduledInstructions,
                            uint32_t numWarps = 1,
                            uint32_t threadsPerWarp = 32);

    // Get instruction latency information
    size_t getInstructionLatency(InstructionTypes type) const;

    // Set instruction latency information
    void setInstructionLatency(InstructionTypes type, size_t cycles);

    // Get register usage information for an instruction
    void getRegisterUsage(const DecodedInstruction& instruction,
                         std::vector<RegisterID>& inputRegisters,
                         std::vector<RegisterID>& outputRegisters) const;

    // Get last scheduling statistics
    const std::unordered_map<std::string, double>& getSchedulingStats() const;

private:
    // Private implementation details
    class Impl;
    std::unique_ptr<Impl> pImpl;
};

#endif // INSTRUCTION_SCHEDULER_HPP