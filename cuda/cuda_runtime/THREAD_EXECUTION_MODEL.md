# PTX VM 多线程执行模型说明

## 🎯 核心问题
当 `cudaLaunchKernel` 启动一个 kernel 时，grid 和 block 配置指定了成百上千的线程，PTX VM 如何模拟这些线程的并行执行？

---

## 🏗️ PTX VM 的执行架构

### SIMT 执行模型 (Single Instruction Multiple Threads)

PTX VM 实现了 GPU 的 SIMT 执行模型，通过以下方式模拟多线程并行：

```
Grid (所有线程)
  └── 多个 Block (CTA - Cooperative Thread Array)
       └── 多个 Warp (32个线程一组)
            └── 32个 Thread 并行执行相同指令
```

### 关键组件

#### 1. **Warp Scheduler (src/execution/warp_scheduler.cpp)**
负责管理和调度 warp 的执行：

```cpp
class WarpScheduler {
    uint32_t m_numWarps;            // warp 数量
    uint32_t m_threadsPerWarp;      // 每个 warp 的线程数（通常32）
    std::vector<Warp*> m_warps;     // 所有 warp 对象
    uint32_t m_currentWarp;         // 当前执行的 warp
}
```

#### 2. **Warp 对象**
表示一组并行执行的线程：

```cpp
class Warp {
    uint32_t m_warpId;              // Warp ID
    uint32_t m_numThreads;          // 线程数量
    uint64_t m_activeMask;          // 活跃线程掩码（64位，每位代表一个线程）
    size_t m_currentPC;             // 当前程序计数器
    std::vector<size_t> m_threadPCs; // 每个线程的 PC（用于分支分歧）
}
```

---

## 🔄 执行流程

### 1. 内核启动 (cudaLaunchKernel)

```cpp
cudaLaunchKernel(func, 
    dim3(blocksPerGrid),      // Grid: 4 blocks
    dim3(threadsPerBlock),    // Block: 256 threads
    args, 0, nullptr);
```

**计算总线程数**：
- Total threads = gridDim × blockDim = 4 × 256 = 1024 threads
- Warps needed = 1024 / 32 = 32 warps

### 2. PTX VM 初始化 (在 HostAPI::cuLaunchKernel 中)

```cpp
// 在 host_api.cpp 中
CUresult cuLaunchKernel(...) {
    // 1. 计算 warp 数量
    uint32_t totalThreads = gridDimX * gridDimY * gridDimZ * 
                            blockDimX * blockDimY * blockDimZ;
    uint32_t numWarps = (totalThreads + 31) / 32;
    
    // 2. 初始化 WarpScheduler
    WarpScheduler& scheduler = executor.getWarpScheduler();
    scheduler.initialize(numWarps, 32);
    
    // 3. 设置每个 warp 的初始状态
    for (uint32_t warpId = 0; warpId < numWarps; ++warpId) {
        scheduler.setActiveThreads(warpId, 0xFFFFFFFF);  // 所有线程激活
        scheduler.setCurrentPC(warpId, 0);               // 从 PC=0 开始
    }
    
    // 4. 执行内核
    bool success = m_vm->run();
}
```

### 3. 执行循环 (在 PTXVM::run 中)

```cpp
bool PTXVM::run() {
    WarpScheduler& scheduler = getWarpScheduler();
    
    // 主执行循环
    while (!scheduler.allWarpsComplete()) {
        // 选择下一个要执行的 warp（轮询调度）
        uint32_t warpId = scheduler.selectNextWarp();
        
        if (!scheduler.warpHasWork(warpId)) {
            continue;
        }
        
        // 获取当前 warp 的状态
        uint64_t activeMask = scheduler.getActiveThreads(warpId);
        size_t currentPC = scheduler.getCurrentPC(warpId);
        
        // 取指令
        DecodedInstruction instr = fetchInstruction(currentPC);
        
        // 执行指令（所有活跃线程并行执行）
        executeInstruction(instr, warpId, activeMask);
        
        // 更新 PC
        scheduler.setCurrentPC(warpId, currentPC + 1);
    }
    
    return true;
}
```

---

## 🧵 线程并行的实现方式

### 方式 1: 逻辑并行（当前实现）

PTX VM 使用 **串行模拟并行** 的方式：

```cpp
void executeInstruction(DecodedInstruction& instr, uint32_t warpId, uint64_t activeMask) {
    RegisterBank& regBank = getRegisterBank();
    
    // 遍历 warp 中的每个线程
    for (uint32_t threadId = 0; threadId < 32; ++threadId) {
        // 检查该线程是否活跃
        if (!(activeMask & (1ULL << threadId))) {
            continue;  // 跳过非活跃线程
        }
        
        // 设置线程 ID（用于 %tid, %ntid 等特殊寄存器）
        regBank.setThreadId(threadId, 0, 0);
        
        // 执行指令（针对这个线程）
        switch (instr.opcode) {
            case PTXOpcode::ADD:
                // 读取源寄存器
                uint64_t src1 = regBank.read(instr.src1, threadId);
                uint64_t src2 = regBank.read(instr.src2, threadId);
                // 执行计算
                uint64_t result = src1 + src2;
                // 写入目标寄存器
                regBank.write(instr.dest, result, threadId);
                break;
            // ... 其他指令
        }
    }
}
```

**关键点**：
- 每个 warp 的 32 个线程 **逐个串行执行**
- 但逻辑上是 **同时执行同一条指令**
- 通过 `activeMask` 控制哪些线程实际执行

### 方式 2: 物理并行（可能的优化）

PTX VM **可以** 使用 C++ 线程池进行真正的并行：

```cpp
void executeInstructionParallel(DecodedInstruction& instr, uint32_t warpId, uint64_t activeMask) {
    std::vector<std::thread> threads;
    
    // 为每个活跃线程创建实际的 OS 线程
    for (uint32_t threadId = 0; threadId < 32; ++threadId) {
        if (activeMask & (1ULL << threadId)) {
            threads.emplace_back([&, threadId]() {
                // 在独立线程中执行
                executeForThread(instr, warpId, threadId);
            });
        }
    }
    
    // 等待所有线程完成
    for (auto& t : threads) {
        t.join();
    }
}
```

**注意**：当前 PTX VM **似乎使用串行模拟**，这对于调试和简单场景已足够。

---

## 🔀 分支分歧 (Divergence) 处理

当 warp 中的线程遇到分支时：

```ptx
@p setp.eq.s32 p, %tid.x, 0    // 只有线程0设置谓词p
@p bra target                   // 只有p为true的线程跳转
```

**处理方式**：

```cpp
void handleBranchDivergence(uint32_t warpId, uint64_t takenMask, 
                           size_t targetPC, size_t fallthroughPC) {
    Warp& warp = getWarp(warpId);
    
    // 1. 保存分歧点到堆栈
    DivergenceStackEntry entry;
    entry.reconvergencePC = fallthroughPC;
    entry.activeMask = ~takenMask;  // 未跳转的线程
    warp.pushDivergence(entry);
    
    // 2. 更新活跃掩码（只执行跳转的线程）
    warp.setActiveMask(takenMask);
    warp.setCurrentPC(targetPC);
    
    // 3. 当跳转路径执行完，恢复到 reconvergencePC
    //    合并两条路径，继续执行
}
```

---

## 📝 在 libptxrt 中的影响

### 当前实现

```cpp
// cuda_runtime.cpp
cudaError_t cudaLaunchKernel(const void* func, dim3 gridDim, dim3 blockDim,
                             void** args, size_t sharedMem, cudaStream_t stream) {
    // ...
    
    // 调用 HostAPI
    CUresult result = state.host_api.cuLaunchKernel(
        f,
        gridDim.x, gridDim.y, gridDim.z,
        blockDim.x, blockDim.y, blockDim.z,
        sharedMem,
        nullptr,
        args,
        nullptr
    );
    
    // PTX VM 内部会：
    // 1. 计算 warp 数量
    // 2. 初始化 WarpScheduler
    // 3. 串行模拟所有 warp 的并行执行
    // 4. 返回结果
}
```

### 需要注意的点

1. **执行是同步的**：`m_vm->run()` 会阻塞直到所有线程完成
2. **没有真正的并行**：线程是逐个执行的（除非 PTX VM 内部使用了线程池）
3. **内存访问是串行的**：不会有真正的竞态条件
4. **性能特征不同**：
   - GPU：真正的并行，受内存带宽限制
   - PTX VM：串行模拟，受 CPU 单核性能限制

---

## 🎓 总结

### PTX VM 如何模拟多线程？

1. **逻辑结构**：
   - Grid → Block → Warp → Thread 的层次结构
   - 每个 Warp 包含 32 个逻辑线程

2. **执行方式**：
   - **串行模拟并行**：一次执行一个 warp 的一条指令
   - 通过 `activeMask` 控制哪些线程执行
   - 通过特殊寄存器（%tid.x, %ntid.x 等）让每个线程知道自己的 ID

3. **调度**：
   - WarpScheduler 使用轮询（round-robin）调度 warp
   - 处理分支分歧和重新汇聚
   - 管理同步原语（__syncthreads）

4. **对 libptxrt 的影响**：
   - `cudaLaunchKernel` 是**同步的**，会等待所有线程完成
   - 不需要额外的线程管理或同步
   - 参数传递给所有线程（通过参数内存）

### 示例执行流程

```
cudaLaunchKernel(kernel, dim3(2), dim3(64), args)
  ↓
totalThreads = 2 × 64 = 128
numWarps = 128 / 32 = 4
  ↓
For each instruction in kernel:
    For warpId = 0 to 3:
        For threadId = 0 to 31:
            if thread is active:
                Execute instruction for this thread
  ↓
All warps complete
  ↓
Return cudaSuccess
```

---

## 🔍 验证方法

可以添加调试日志来观察执行：

```cpp
// 在 cuda_runtime.cpp 中
cudaError_t cudaLaunchKernel(...) {
    printf("[libptxrt] Total threads: %u × %u × %u = %u\n",
           gridDim.x * blockDim.x, gridDim.y * blockDim.y, gridDim.z * blockDim.z,
           gridDim.x * blockDim.x * gridDim.y * blockDim.y * gridDim.z * blockDim.z);
    printf("[libptxrt] Estimated warps: %u\n",
           (gridDim.x * blockDim.x * gridDim.y * blockDim.y * gridDim.z * blockDim.z + 31) / 32);
    
    // ... 调用 HostAPI
}
```

这样运行 simple_add 时就能看到：
```
[libptxrt] Total threads: 1024
[libptxrt] Estimated warps: 32
[libptxrt] Launching kernel...
```

---

**参考文档**：
- `src/execution/warp_scheduler.cpp` - Warp 调度实现
- `docs_spec/warp_scheduler.md` - Warp 调度器规范
- `src/host/host_api.cpp` - 内核启动实现
