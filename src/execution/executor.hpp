#ifndef PTX_EXECUTOR_HPP
#define PTX_EXECUTOR_HPP

#include <vector>
#include <memory>
#include "parser/parser.hpp"
#include "decoder/decoder.hpp"
#include "registers/register_bank.hpp"
#include "memory/memory.hpp"
#include "include/instruction_types.hpp"
#include "warp_scheduler.hpp"
#include "predicate_handler.hpp"
#include "reconvergence_mechanism.hpp"

// Forward declarations
class CFGNode;
class ControlFlowGraph;

// Divergence handling types
struct DivergenceStackEntry {
    size_t joinPC;          // Program counter at divergence point
    uint64_t activeMask;    // Active threads mask before divergence
    uint64_t divergentMask; // Threads that took the branch
    bool isJoinPointValid;  // Whether the join point is valid
};

typedef std::vector<DivergenceStackEntry> DivergenceStack;

// Divergence statistics
struct DivergenceStats {
    size_t numDivergentPaths;         // Total number of divergent paths
    size_t maxDivergenceDepth;        // Maximum depth of divergence stack
    double averageDivergenceRate;     // Average rate of divergence
    double averageReconvergenceTime;  // Average time to reconverge
    double divergenceImpactFactor;    // Impact on performance
};

// Reconvergence algorithms
enum ReconvergenceAlgorithm {
    RECONVERGENCE_ALGORITHM_BASIC,
    RECONVERGENCE_ALGORITHM_CFG_BASED,
    RECONVERGENCE_ALGORITHM_STACK_BASED
};

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
    const std::vector<CFGNode*>& getPredecessors() const { return m_predecessors; }
    
private:
    size_t m_pc;  // Program counter for this node
    
    // Control flow graph connections
    std::vector<CFGNode*> m_successors;
    std::vector<CFGNode*> m_predecessors;
    
    // Post-dominator information
    std::unordered_set<CFGNode*> m_immediatePostDominators;
    
    // Reconvergence point
    size_t m_reconvergencePC;
    bool m_hasReconvergence;
};

// Control flow graph
class ControlFlowGraph {
public:
    ControlFlowGraph();
    ~ControlFlowGraph();
    
    // Build CFG from PTX code
    bool buildFromPTX(const std::string& ptxCode);
    
    // Get node for PC
    CFGNode* getNodeForPC(size_t pc);
    
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

// Add synchronization support
enum class SyncType {
    SYNC_UNDEFINED,  // Undefined or unsupported synchronization type
    SYNC_WARP,       // Warp-level synchronization
    SYNC_CTA,        // CTA-level synchronization
    SYNC_GRID,       // Grid-level synchronization
    SYNC_MEMBAR      // Memory barrier
};

struct DecodedInstruction {
    // ... existing members ...
    
    // Synchronization information
    SyncType syncType = SyncType::SYNC_UNDEFINED;
    uint32_t syncScope = 0;  // Scope of synchronization (CTA ID, Grid ID, etc.)
    
    // ... existing members ...
};

class Executor {
public:
    // Constructor/destructor
    Executor();
    ~Executor();

    // Initialize the executor with parsed instructions
    bool initialize(const std::vector<PTXInstruction>& ptInstructions);

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
    RegisterBank& getRegisterBank() {
        return *m_registerBank;
    }
    
    MemorySubsystem& getMemorySubsystem() {
        return *m_memorySubsystem;
    }
    
    // Get reference to warp scheduler
    WarpScheduler& getWarpScheduler() {
        return *m_warpScheduler;
    }

    // Get current CTA and Grid IDs
    uint32_t getCurrentCtaId() const { return m_currentCtaId; }
    uint32_t getCurrentGridId() const { return m_currentGridId; }
    
    // Get divergence statistics
    const DivergenceStats& getDivergenceStats() const {
        return m_divergenceStats;
    }
    
    // Set reconvergence algorithm
    void setReconvergenceAlgorithm(ReconvergenceAlgorithm algorithm) {
        m_reconvergenceAlgorithm = algorithm;
    }
    
private:
    // Private implementation details
    class Impl;
    std::unique_ptr<Impl> pImpl;
    
    // Core components
    std::unique_ptr<RegisterBank> m_registerBank;
    std::unique_ptr<MemorySubsystem> m_memorySubsystem;
    std::unique_ptr<WarpScheduler> m_warpScheduler;
    std::unique_ptr<PredicateHandler> m_predicateHandler;
    
    // Divergence and reconvergence handling
    std::unique_ptr<ReconvergenceMechanism> m_reconvergence;
    
    // Performance counters
    PerformanceCounters& m_performanceCounters;
    
    // Current execution context
    uint32_t m_currentCtaId = 0;
    uint32_t m_currentGridId = 0;
    
    // Divergence handling
    DivergenceStack m_divergenceStack;
    size_t m_divergenceStartCycle = 0;
    size_t m_numDivergences = 0;
    DivergenceStats m_divergenceStats;
    ReconvergenceAlgorithm m_reconvergenceAlgorithm = RECONVERGENCE_ALGORITHM_BASIC;
    
    // Control flow graph
    ControlFlowGraph m_controlFlowGraph;
    std::unordered_map<size_t, CFGNode*> m_pcToNode;
    
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
    
};

#endif // PTX_EXECUTOR_HPP