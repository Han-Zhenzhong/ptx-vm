#include "warp_scheduler.hpp"
#include <iostream>
#include <stdexcept>

// Private implementation class
class WarpScheduler::Impl {
public:
    Impl(uint32_t numWarps, uint32_t threadsPerWarp) : 
        m_numWarps(numWarps), 
        m_threadsPerWarp(threadsPerWarp) {
        // Create warps
        for (uint32_t i = 0; i < m_numWarps; ++i) {
            m_warps.push_back(std::make_unique<Warp>(i, m_threadsPerWarp));
        }
    }
    
    ~Impl() = default;
    
    // Initialize the scheduler
    bool initialize() {
        // Initialize all warps
        for (const auto& warp : m_warps) {
            // Start with all threads active
            warp->setActiveMask((1ULL << m_threadsPerWarp) - 1);
            
            // Initial PC is zero
            warp->setCurrentPC(0);
        }
        
        // Reset current warp selection
        m_currentWarp = 0;
        
        return true;
    }
    
    // Reset the scheduler for new kernel launch
    void reset() {
        // Reset all warps
        for (const auto& warp : m_warps) {
            // Start with all threads active
            warp->setActiveMask((1ULL << m_threadsPerWarp) - 1);
            
            // Initial PC is zero
            warp->setCurrentPC(0);
            
            // Clear divergence stack
            while (!warp->isDivergenceStackEmpty()) {
                warp->popDivergencePoint();
            }
        }
        
        // Reset current warp selection
        m_currentWarp = 0;
    }
    
    // Get number of warps
    uint32_t getNumWarps() const {
        return m_numWarps;
    }
    
    // Get number of threads per warp
    uint32_t getThreadsPerWarp() const {
        return m_threadsPerWarp;
    }
    
    // Get current warp being executed
    uint32_t getCurrentWarp() const {
        return m_currentWarp;
    }
    
    // Get active threads in a warp
    uint64_t getActiveThreads(uint32_t warpId) const {
        if (warpId >= m_numWarps) {
            return 0;
        }
        return m_warps[warpId]->getActiveMask();
    }
    
    // Set active threads in a warp
    void setActiveThreads(uint32_t warpId, uint64_t activeMask) {
        if (warpId < m_numWarps) {
            m_warps[warpId]->setActiveMask(activeMask);
        }
    }
    
    // Get current PC for a warp
    size_t getCurrentPC(uint32_t warpId) const {
        if (warpId < m_numWarps) {
            return m_warps[warpId]->getCurrentPC();
        }
        return 0;
    }
    
    // Set current PC for a warp
    void setCurrentPC(uint32_t warpId, size_t pc) {
        if (warpId < m_numWarps) {
            m_warps[warpId]->setCurrentPC(pc);
        }
    }
    
    // Get next PC for a warp
    size_t getNextPC(uint32_t warpId) const {
        if (warpId < m_numWarps) {
            return m_warps[warpId]->getNextPC();
        }
        return 0;
    }
    
    // Set next PC for a warp
    void setNextPC(uint32_t warpId, size_t pc) {
        if (warpId < m_numWarps) {
            m_warps[warpId]->setNextPC(pc);
        }
    }
    
    // Select next warp to execute
    uint32_t selectNextWarp() {
        // Simple round-robin scheduling for now
        uint32_t originalWarp = m_currentWarp;
        
        do {
            // Move to next warp
            m_currentWarp = (m_currentWarp + 1) % m_numWarps;
            
            // Check if this warp has work
            if (warpHasWork(m_currentWarp)) {
                return m_currentWarp;
            }
        } while (m_currentWarp != originalWarp);
        
        // No warps have work
        return m_numWarps;  // Invalid warp ID
    }
    
    // Issue instruction from current warp
    bool issueInstruction(InstructionIssueInfo& issueInfo) {
        uint32_t warpId = m_currentWarp;
        
        if (warpId >= m_numWarps) {
            return false;
        }
        
        // Get current PC for this warp
        size_t currentPC = m_warps[warpId]->getCurrentPC();
        
        // Check if there's valid instruction at this PC
        // This would normally check against actual instruction count
        if (currentPC >= 0) {  // Simplified for now
            issueInfo.warpId = warpId;
            issueInfo.instructionIndex = static_cast<uint32_t>(currentPC);
            issueInfo.activeMask = m_warps[warpId]->getActiveMask();
            
            // For now, assume we can always issue an instruction
            return true;
        }
        
        return false;
    }
    
    // Complete execution of an instruction
    void completeInstruction(const InstructionIssueInfo& issueInfo) {
        uint32_t warpId = issueInfo.warpId;
        
        if (warpId >= m_numWarps) {
            return;
        }
        
        // In real implementation, this would update warp state after instruction completes
        // For now, just advance PC by one
        size_t nextPC = m_warps[warpId]->getCurrentPC() + 1;
        m_warps[warpId]->setCurrentPC(nextPC);
    }
    
    // Handle branch divergence
    void handleBranchDivergence(uint32_t warpId, 
                               uint64_t takenMask,
                               size_t targetPC,
                               size_t fallthroughPC) {
        if (warpId >= m_numWarps) {
            return;
        }
        
        // Save divergence point
        // This is simplified - real implementation would be more complex
        if (takenMask != 0 && takenMask != m_warps[warpId]->getActiveMask()) {
            // Some threads take branch, some don't - need to reconverge
            m_warps[warpId]->pushDivergencePoint(fallthroughPC);
            
            // Update active mask for threads that took the branch
            m_warps[warpId]->setActiveMask(takenMask);
            
            // Set PC to branch target for these threads
            m_warps[warpId]->setCurrentPC(targetPC);
        }
    }
    
    // Check if warp has work
    bool warpHasWork(uint32_t warpId) const {
        if (warpId >= m_numWarps) {
            return false;
        }
        
        // For now, assume warp has work unless all threads are inactive
        return m_warps[warpId]->anyActive();
    }
    
    // Check if all warps have completed execution
    bool allWarpsComplete() const {
        for (const auto& warp : m_warps) {
            if (warp->anyActive()) {
                return false;
            }
        }
        return true;
    }
    
private:
    // Core configuration
    uint32_t m_numWarps;
    uint32_t m_threadsPerWarp;
    uint32_t m_currentWarp;
    
    // Warps
    std::vector<std::unique_ptr<Warp>> m_warps;
};

WarpScheduler::WarpScheduler(uint32_t numWarps, uint32_t threadsPerWarp) : 
    pImpl(std::make_unique<Impl>(numWarps, threadsPerWarp)) {}

WarpScheduler::~WarpScheduler() = default;

bool WarpScheduler::initialize() {
    return pImpl->initialize();
}

void WarpScheduler::reset() {
    pImpl->reset();
}

uint32_t WarpScheduler::getNumWarps() const {
    return pImpl->getNumWarps();
}

uint32_t WarpScheduler::getThreadsPerWarp() const {
    return pImpl->getThreadsPerWarp();
}

uint32_t WarpScheduler::getCurrentWarp() const {
    return pImpl->getCurrentWarp();
}

uint64_t WarpScheduler::getActiveThreads(uint32_t warpId) const {
    return pImpl->getActiveThreads(warpId);
}

void WarpScheduler::setActiveThreads(uint32_t warpId, uint64_t activeMask) {
    pImpl->setActiveThreads(warpId, activeMask);
}

size_t WarpScheduler::getCurrentPC(uint32_t warpId) const {
    return pImpl->getCurrentPC(warpId);
}

void WarpScheduler::setCurrentPC(uint32_t warpId, size_t pc) {
    pImpl->setCurrentPC(warpId, pc);
}

size_t WarpScheduler::getNextPC(uint32_t warpId) const {
    return pImpl->getNextPC(warpId);
}

void WarpScheduler::setNextPC(uint32_t warpId, size_t pc) {
    pImpl->setNextPC(warpId, pc);
}

uint32_t WarpScheduler::selectNextWarp() {
    return pImpl->selectNextWarp();
}

bool WarpScheduler::issueInstruction(InstructionIssueInfo& issueInfo) {
    return pImpl->issueInstruction(issueInfo);
}

void WarpScheduler::completeInstruction(const InstructionIssueInfo& issueInfo) {
    pImpl->completeInstruction(issueInfo);
}

void WarpScheduler::handleBranchDivergence(uint32_t warpId, 
                                           uint64_t takenMask,
                                           size_t targetPC,
                                           size_t fallthroughPC) {
    pImpl->handleBranchDivergence(warpId, takenMask, targetPC, fallthroughPC);
}

bool WarpScheduler::warpHasWork(uint32_t warpId) const {
    return pImpl->warpHasWork(warpId);
}

bool WarpScheduler::allWarpsComplete() const {
    return pImpl->allWarpsComplete();
}

// CTA-level synchronization
bool WarpScheduler::syncThreadsInCta(uint32_t ctaId, size_t syncPC) {
    // Get current warp
    uint32_t currentWarpId = m_currentWarp;
    
    // Check if this is the first warp to reach the synchronization point
    if (m_syncState.find(ctaId) == m_syncState.end()) {
        // Initialize sync state for this CTA
        SyncState state;
        state.ctaId = ctaId;
        state.syncPC = syncPC;
        state.arrivalMask = 0;
        state.syncCount = 0;
        m_syncState[ctaId] = state;
    }
    
    SyncState& state = m_syncState[ctaId];
    
    // Mark this warp as arrived
    state.arrivalMask |= (1 << currentWarpId);
    
    // Increment sync count
    state.syncCount++;
    
    // Check if all warps in CTA have reached the synchronization point
    if (checkCtaThreadsCompleted(ctaId)) {
        // All threads have reached the synchronization point, continue execution
        // Reset sync state
        state.arrivalMask = 0;
        state.syncCount = 0;
        
        return true;
    }
    
    // Not all threads have arrived yet, return false
    return false;
}

// Grid-level synchronization
bool WarpScheduler::syncThreadsInGrid(uint32_t gridId, size_t syncPC) {
    // Get current warp
    uint32_t currentWarpId = m_currentWarp;
    
    // Check if this is the first warp to reach the synchronization point
    if (m_gridSyncState.find(gridId) == m_gridSyncState.end()) {
        // Initialize sync state for this grid
        SyncState state;
        state.gridId = gridId;
        state.syncPC = syncPC;
        state.arrivalMask = 0;
        state.syncCount = 0;
        m_gridSyncState[gridId] = state;
    }
    
    SyncState& state = m_gridSyncState[gridId];
    
    // Mark this warp as arrived
    state.arrivalMask |= (1 << currentWarpId);
    
    // Increment sync count
    state.syncCount++;
    
    // Check if all CTAs in grid have reached the synchronization point
    if (checkGridCtasCompleted(gridId)) {
        // All CTAs have reached the synchronization point, continue execution
        // Reset sync state
        state.arrivalMask = 0;
        state.syncCount = 0;
        
        return true;
    }
    
    // Not all CTAs have arrived yet, return false
    return false;
}

// Check if all threads in CTA have reached synchronization point
bool WarpScheduler::checkCtaThreadsCompleted(uint32_t ctaId) {
    // Implementation specific to CTA thread tracking
    // This is a simplified implementation
    if (m_syncState.find(ctaId) == m_syncState.end()) {
        return false;
    }
    
    const SyncState& state = m_syncState[ctaId];
    
    // For simplicity, we assume all threads complete when all warps have synced
    // In a real implementation, this would check the actual number of warps
    // in the CTA against the sync count
    return state.syncCount >= WARPS_PER_CTA;
}

// Check if all CTAs in grid have reached synchronization point
bool WarpScheduler::checkGridCtasCompleted(uint32_t gridId) {
    // Implementation specific to grid CTA tracking
    // This is a simplified implementation
    if (m_gridSyncState.find(gridId) == m_gridSyncState.end()) {
        return false;
    }
    
    const SyncState& state = m_gridSyncState[gridId];
    
    // For simplicity, we assume all CTAs complete when all CTAs have synced
    // In a real implementation, this would check the actual number of CTAs
    // in the grid against the sync count
    return state.syncCount >= CTAS_PER_GRID;
}

// Implementation of Warp class methods
Warp::Warp(uint32_t warpId, uint32_t numThreads) :
    m_warpId(warpId), 
    m_numThreads(numThreads),
    m_activeMask(0),
    m_currentPC(0),
    m_nextPC(0),
    m_divergenceStackDepth(0) {
    // Allocate thread program counters
    if (m_numThreads > 0 && m_numThreads <= 64) {
        m_threadPCs.resize(m_numThreads);
        for (uint32_t i = 0; i < m_numThreads; ++i) {
            m_threadPCs[i] = m_currentPC;
        }
    }
}

Warp::~Warp() = default;

bool Warp::allActive() const {
    // Check if all threads are active
    if (m_numThreads == 0 || m_numThreads > 64) {
        return false;
    }
    
    // All bits up to m_numThreads should be set
    uint64_t expectedMask = (1ULL << m_numThreads) - 1;
    return (m_activeMask & expectedMask) == expectedMask;
}

bool Warp::anyActive() const {
    // Check if any thread is active
    return (m_activeMask != 0);
}

size_t Warp::getThreadPC(uint32_t threadId) const {
    if (threadId < m_numThreads) {
        return m_threadPCs[threadId];
    }
    return 0;
}

void Warp::setThreadPC(uint32_t threadId, size_t pc) {
    if (threadId < m_numThreads) {
        m_threadPCs[threadId] = pc;
        
        // If all threads have same PC, update warp PC
        bool allSame = true;
        size_t firstPC = m_threadPCs[0];
        
        for (uint32_t i = 1; i < m_numThreads; ++i) {
            if (m_threadPCs[i] != firstPC) {
                allSame = false;
                break;
            }
        }
        
        if (allSame) {
            m_currentPC = firstPC;
        }
    }
}

void Warp::pushDivergencePoint(size_t joinPC) {
    m_divergenceStack.push_back(joinPC);
    m_divergenceStackDepth++;
}

size_t Warp::popDivergencePoint() {
    if (m_divergenceStack.empty()) {
        return 0;  // No divergence point to pop
    }
    
    size_t top = m_divergenceStack.back();
    m_divergenceStack.pop_back();
    m_divergenceStackDepth--;
    
    return top;
}

bool Warp::isDivergenceStackEmpty() const {
    return m_divergenceStack.empty();
}