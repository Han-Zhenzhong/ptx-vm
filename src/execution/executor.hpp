#ifndef PTX_EXECUTOR_HPP
#define PTX_EXECUTOR_HPP

#include <vector>
#include <memory>
#include "parser/parser.hpp"
#include "decoder/decoder.hpp"
#include "registers/register_bank.hpp"
#include "memory/memory.hpp"
#include "instruction_types.hpp"
#include "warp_scheduler.hpp"
#include "predicate_handler.hpp"
#include "reconvergence_mechanism.hpp"
#include <unordered_set>
#include "performance_counters.hpp"
#include "optimizer/instruction_scheduler.hpp"

// Forward declarations
class CFGNode;
class ControlFlowGraph;

// Control flow graph node
class CFGNode {
public:
    CFGNode(size_t pc);
    ~CFGNode();
    
    // Add a successor node
    void addSuccessor(CFGNode* node);
    
    // Add a predecessor node
    void addPredecessor(CFGNode* node);
    
    // Get immediate post-dominators
    const std::unordered_set<CFGNode*>& getImmediatePostDominators() const;
    
    // Check if this node has a reconvergence point
    bool hasReconvergence() const;
    
    // Get reconvergence PC
    size_t getReconvergencePC() const;
    
    // Set reconvergence point
    void setReconvergencePoint(size_t pc);
    
    // Add an immediate post-dominator
    void addImmediatePostDominator(CFGNode* node);
    
    // Get PC
    size_t getPC() const { return m_pc; }
    
    // Get predecessors
    const std::unordered_set<CFGNode*>& getPredecessors() const { return m_predecessors; }
    
    // Get successors
    const std::unordered_set<CFGNode*>& getSuccessors() const { return m_successors; }

private:
    size_t m_pc;  // Program counter value for this node
    std::unordered_set<CFGNode*> m_predecessors;  // Predecessor nodes
    std::unordered_set<CFGNode*> m_successors;    // Successor nodes
    std::unordered_set<CFGNode*> m_immediatePostDominators;  // Immediate post-dominators
    size_t m_reconvergencePC = 0;  // Reconvergence point PC
    bool m_hasReconvergence = false;  // Whether this node has a reconvergence point
};

// Control flow graph
class ControlFlowGraph {
public:
    ControlFlowGraph();
    ~ControlFlowGraph();
    
    // Add a node to the graph
    void addNode(CFGNode* node);
    
    // Get node by PC
    CFGNode* getNode(size_t pc);
    
    // Build CFG from instructions
    bool buildFromInstructions(const std::vector<DecodedInstruction>& instructions);
    
    // Calculate immediate post-dominators for all nodes
    void calculateImmediatePostDominators();
    
    // Find reconvergence points
    void findReconvergencePoints();
    
    // Get reconvergence point for PC
    size_t getReconvergencePC(size_t pc);
    
private:
    // Calculate immediate post-dominators for a node
    void calculateImmediatePostDominators(CFGNode* node);
    
    // Map from PC to CFG node
    std::unordered_map<size_t, CFGNode*> m_pcToNode;
};

class PTXExecutor {
public:
    // Constructor/destructor
    PTXExecutor(RegisterBank& registerBank, MemorySubsystem& memorySubsystem, PerformanceCounters& performanceCounters);
    ~PTXExecutor();

    // Initialize the executor with parsed instructions
    bool initialize(const std::vector<PTXInstruction>& ptInstructions);
    
    // Initialize the executor with decoded instructions directly
    bool initialize(const std::vector<DecodedInstruction>& decodedInstructions);
    
    // Initialize the executor with complete PTX program (recommended)
    bool initialize(const PTXProgram& program);

    // Execute the program
    bool execute();

    // Execute a single instruction
    bool executeSingleInstruction();

    // Get current instruction index
    size_t getCurrentInstructionIndex() const;

    // Check if execution complete
    bool isExecutionComplete() const;

    // Get decoded instructions
    const std::vector<DecodedInstruction>& getDecodedInstructions() const;

    // Get references to core components
    RegisterBank& getRegisterBank();
    
    MemorySubsystem& getMemorySubsystem();
    
    // Get reference to warp scheduler
    WarpScheduler& getWarpScheduler();
    
    // Get reference to predicate handler
    PredicateHandler& getPredicateHandler();
    
    // Get reference to reconvergence mechanism
    ReconvergenceMechanism& getReconvergenceMechanism();
    
    // Get reference to instruction scheduler
    InstructionScheduler& getInstructionScheduler();
    
    // Get reference to performance counters
    PerformanceCounters& getPerformanceCounters();
    
    // ========================================================================
    // Multi-function execution support (requires PTXProgram initialization)
    // ========================================================================
    
    // Call a function by name with arguments
    bool callFunction(const std::string& funcName, const std::vector<uint64_t>& args = {});
    
    // Check if program structure is available
    bool hasProgramStructure() const;
    
    // Get current call stack depth
    size_t getCallStackDepth() const;
    
    // Get the loaded PTX program structure
    const PTXProgram& getProgram() const;

private:
    // Private implementation details
    class Impl;
    std::unique_ptr<Impl> pImpl;
    
    // Synchronization support methods
    void handleSynchronization(const DecodedInstruction& instruction);
    void handleCtaSynchronization();
    void handleGridSynchronization();
    void handleMemoryBarrier();
    bool checkCtaThreadsCompleted(uint32_t ctaId);
    bool checkGridCtasCompleted(uint32_t gridId);
    void flushMemoryCaches();
    
    // Divergence handling methods
    void handleDivergence(const DecodedInstruction& instruction, uint64_t activeMask, uint64_t threadMask);
    void basicReconvergence(const DecodedInstruction& instruction, uint64_t& activeMask, bool takeBranch, uint64_t threadMask);
    void cfgBasedReconvergence(const DecodedInstruction& instruction, uint64_t& activeMask, bool takeBranch, uint64_t threadMask);
    void stackBasedReconvergence(const DecodedInstruction& instruction, uint64_t& activeMask, bool takeBranch, uint64_t threadMask);
    void updateDivergenceStats(bool takeBranch);
    void updateReconvergenceStats(size_t divergenceCycles);
    size_t findCFGReconvergencePoint(size_t pc);
    bool buildControlFlowGraphFromPTX(const std::vector<DecodedInstruction>& instructions);
    
    // Parameter instruction execution methods
    bool executeLDParam(const DecodedInstruction& instr);
    bool executeSTParam(const DecodedInstruction& instr);
    
};

#endif // PTX_EXECUTOR_HPP