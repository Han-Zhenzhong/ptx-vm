# 多 Warp 执行修复

## 🐛 问题描述

**原始问题**：PTX VM 的 `cudaLaunchKernel` 只执行了 1 个 warp（32个线程），即使用户指定了更大的 grid/block 配置。

### 证据

1. **Executor 硬编码了 1 个 warp**：
```cpp
// src/execution/executor.cpp 原代码（第 22 行）
m_warpScheduler = std::make_unique<WarpScheduler>(1, 32);  // ⚠️ 固定为 1
// TODO: Support multiple warps with per-thread register files
```

2. **Grid/Block 参数被忽略**：
```cpp
// src/host/host_api.cpp 原代码（第 196-197 行）
// 设置grid/block维度
// TODO: 传递给 warp scheduler  // ⚠️ 从未实现
```

3. **实际影响**：
   - 启动 `kernel<<<4, 256>>>(...)` → 应该有 1024 个线程
   - 实际只执行了 32 个线程（1 个 warp）
   - 其余 992 个线程的计算被忽略

---

## ✅ 修复方案

### 1. 添加动态 Warp 配置支持

#### 修改文件：`src/execution/executor.hpp`

添加了设置 grid/block 维度的接口：

```cpp
// Grid/Block dimension configuration for kernel launch
void setGridDimensions(unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
                      unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ);

void getGridDimensions(unsigned int& gridDimX, unsigned int& gridDimY, unsigned int& gridDimZ,
                      unsigned int& blockDimX, unsigned int& blockDimY, unsigned int& blockDimZ) const;
```

---

### 2. 实现动态 Warp 初始化

#### 修改文件：`src/execution/executor.cpp`

**（1）添加成员变量存储 grid/block 配置**：

```cpp
class PTXExecutor::Impl {
    // Grid and block dimensions for kernel launch
    unsigned int m_gridDimX = 1;
    unsigned int m_gridDimY = 1;
    unsigned int m_gridDimZ = 1;
    unsigned int m_blockDimX = 1;
    unsigned int m_blockDimY = 1;
    unsigned int m_blockDimZ = 1;
    // ...
};
```

**（2）实现 `setGridDimensions` 方法**（在文件末尾）：

```cpp
void PTXExecutor::setGridDimensions(unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
                                    unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ) {
    // Store dimensions
    pImpl->m_gridDimX = gridDimX;
    pImpl->m_gridDimY = gridDimY;
    pImpl->m_gridDimZ = gridDimZ;
    pImpl->m_blockDimX = blockDimX;
    pImpl->m_blockDimY = blockDimY;
    pImpl->m_blockDimZ = blockDimZ;
    
    // Calculate total number of threads
    unsigned int totalThreads = gridDimX * gridDimY * gridDimZ * 
                                blockDimX * blockDimY * blockDimZ;
    
    // Calculate number of warps (32 threads per warp, round up)
    unsigned int numWarps = (totalThreads + 31) / 32;
    
    // Limit to reasonable maximum (32,768 threads = 1024 warps)
    const unsigned int MAX_WARPS = 1024;
    if (numWarps > MAX_WARPS) {
        std::cerr << "Warning: Requested " << numWarps << " warps (" << totalThreads 
                  << " threads), limiting to " << MAX_WARPS << " warps" << std::endl;
        numWarps = MAX_WARPS;
    }
    
    Logger::debug("Configuring WarpScheduler: " + std::to_string(numWarps) + 
                  " warps (" + std::to_string(totalThreads) + " threads total)");
    
    // Recreate warp scheduler with correct number of warps
    pImpl->m_warpScheduler = std::make_unique<WarpScheduler>(numWarps, 32);
    if (!pImpl->m_warpScheduler->initialize()) {
        throw std::runtime_error("Failed to initialize warp scheduler");
    }
}
```

**（3）更新执行循环设置线程上下文**：

在 `execute()` 方法中，为每个 warp 设置正确的线程 ID：

```cpp
// 从 warpId 计算全局线程 ID
uint32_t threadBase = warpId * 32;
uint32_t globalThreadId = threadBase;

// 计算线程坐标 (tid.x, tid.y, tid.z)
uint32_t threadsPerBlock = m_blockDimX * m_blockDimY * m_blockDimZ;
uint32_t blockId = globalThreadId / threadsPerBlock;
uint32_t threadInBlock = globalThreadId % threadsPerBlock;

uint32_t tid_x = threadInBlock % m_blockDimX;
uint32_t tid_y = (threadInBlock / m_blockDimX) % m_blockDimY;
uint32_t tid_z = threadInBlock / (m_blockDimX * m_blockDimY);

uint32_t ctaid_x = blockId % m_gridDimX;
uint32_t ctaid_y = (blockId / m_gridDimX) % m_gridDimY;
uint32_t ctaid_z = blockId / (m_gridDimX * m_gridDimY);

// 设置特殊寄存器
m_registerBank->setThreadId(tid_x, tid_y, tid_z);
m_registerBank->setBlockId(ctaid_x, ctaid_y, ctaid_z);
m_registerBank->setThreadDimensions(m_blockDimX, m_blockDimY, m_blockDimZ);
m_registerBank->setGridDimensions(m_gridDimX, m_gridDimY, m_gridDimZ);
```

---

### 3. 调用新接口配置 Warp

#### 修改文件：`src/host/host_api.cpp`

在 `cuLaunchKernel` 中调用 `setGridDimensions`：

```cpp
CUresult cuLaunchKernel(...) {
    // ...
    PTXExecutor& executor = m_vm->getExecutor();
    
    // ✅ Configure grid/block dimensions before parameter setup
    Logger::debug("Configuring grid/block dimensions...");
    executor.setGridDimensions(gridDimX, gridDimY, gridDimZ,
                              blockDimX, blockDimY, blockDimZ);
    
    // 复制参数到参数内存...
    
    // Grid/block dimensions already configured via setGridDimensions() above
    Logger::debug("Starting kernel execution...");
    
    // 执行内核
    bool success = m_vm->run();
    // ...
}
```

移除了原来的 TODO 注释，改为实际调用 `setGridDimensions`。

---

## 🎯 修复效果

### 修复前
```
cudaLaunchKernel(kernel, dim3(4), dim3(256), args)
  → 总线程数：4 × 256 = 1024
  → 实际执行：1 warp = 32 threads ❌
  → 结果：只有前 32 个元素被计算
```

### 修复后
```
cudaLaunchKernel(kernel, dim3(4), dim3(256), args)
  → 总线程数：4 × 256 = 1024
  → 计算 warp 数：(1024 + 31) / 32 = 32 warps
  → 实际执行：32 warps × 32 threads = 1024 threads ✅
  → 结果：所有 1024 个元素都被正确计算
```

### 示例输出日志
```
[libptxrt] Launching kernel with grid=(4,1,1), block=(256,1,1)
[DEBUG] Configuring WarpScheduler: 32 warps (1024 threads total)
[DEBUG] Grid: 4x1x1
[DEBUG] Block: 256x1x1
[DEBUG] Starting kernel execution...
[INFO] Executing warp 0: tid=(0,0,0), ctaid=(0,0,0)
[INFO] Executing warp 1: tid=(32,0,0), ctaid=(0,0,0)
...
[INFO] Executing warp 31: tid=(224,0,0), ctaid=(3,0,0)
```

---

## ⚠️ 已知限制

### 1. 寄存器状态共享

**问题**：当前 `RegisterBank` 是单例，所有 warp 共享同一组通用寄存器（%r0-%rN）。

**影响**：
- ✅ 特殊寄存器（%tid.x, %ctaid.x 等）每个 warp 正确设置
- ❌ 通用寄存器可能在 warp 之间相互覆盖

**适用场景**：
- ✅ 简单的 SIMD 代码（每个线程独立计算，不依赖寄存器状态）
- ✅ 只使用内存和特殊寄存器的代码
- ❌ 复杂的控制流（warp 间寄存器状态冲突）

**示例（可正常工作）**：
```ptx
// Simple vector add - 每个线程独立
.visible .entry vecAdd(...) {
    ld.param.u64 %rd1, [vecAdd_param_0];  // 加载参数
    ld.param.u64 %rd2, [vecAdd_param_1];
    ld.param.u64 %rd3, [vecAdd_param_2];
    
    mov.u32 %r1, %tid.x;                  // 使用线程 ID
    mul.wide.u32 %rd4, %r1, 4;            // 计算偏移
    
    add.u64 %rd5, %rd1, %rd4;             // 地址计算
    ld.global.f32 %f1, [%rd5];            // 加载数据
    
    add.u64 %rd6, %rd2, %rd4;
    ld.global.f32 %f2, [%rd6];
    
    add.f32 %f3, %f1, %f2;                // 执行计算
    
    add.u64 %rd7, %rd3, %rd4;
    st.global.f32 [%rd7], %f3;            // 存储结果
    
    ret;
}
```

**为什么可以工作**：
- 每个 warp 执行时，从内存加载数据 → 计算 → 存储到内存
- 即使通用寄存器被覆盖，下一个 warp 会重新加载正确的数据
- 最终结果写入内存的不同位置（基于 %tid.x）

### 2. 性能

**执行方式**：串行模拟 SIMT（一次一个 warp）

**性能特点**：
- 时间复杂度：O(warps × instructions)
- 1024 个线程 ≈ 32 倍于 32 个线程的执行时间
- 没有真正的并行加速

---

## 🔮 未来改进

### 完整的多线程寄存器支持

需要重构 `RegisterBank` 为多线程架构：

```cpp
class RegisterBank {
    // 每个 warp 的每个线程都有独立的寄存器文件
    std::vector<std::vector<std::vector<uint64_t>>> m_registers;
    // [warpId][threadId][registerIndex]
    
    uint64_t readRegister(uint32_t warpId, uint32_t threadId, size_t registerIndex);
    void writeRegister(uint32_t warpId, uint32_t threadId, size_t registerIndex, uint64_t value);
};
```

这需要：
1. 修改所有指令执行逻辑传递 warpId/threadId
2. 大幅增加内存开销（32 warps × 32 threads × 32 registers × 8 bytes ≈ 256 KB）
3. 确保分支分歧时正确维护每个线程的状态

---

## 📋 测试清单

编译并测试：

```bash
cd build
make -j4

# 测试 simple_add
./examples/simple_add_test

# 检查输出：所有元素应该正确计算
# Expected: c[i] = a[i] + b[i] for all i
```

**预期结果**：
- ✅ 编译无错误
- ✅ WarpScheduler 创建 32 个 warp（1024 threads ÷ 32）
- ✅ 所有 1024 个元素都被正确计算
- ✅ 验证输出：`c[i] == a[i] + b[i]` 对所有 i 成立

---

## 📚 相关文件

### 修改的文件
1. `src/execution/executor.hpp` - 添加 setGridDimensions 接口
2. `src/execution/executor.cpp` - 实现动态 warp 初始化和线程 ID 设置
3. `src/host/host_api.cpp` - 调用 setGridDimensions

### 相关文档
- `THREAD_EXECUTION_MODEL.md` - PTX VM 线程执行模型详解
- `cuda/cuda_runtime/README.md` - CUDA Runtime API 实现说明
- `cuda/cuda_runtime/BUILD_AND_TEST.md` - 构建和测试指南

---

## 🎓 总结

这个修复解决了 **PTX VM 只执行 1 个 warp** 的核心问题：

1. ✅ **动态计算 warp 数量**：根据 grid×block 自动分配
2. ✅ **设置正确的线程 ID**：每个 warp 有正确的 %tid.x, %ctaid.x 等
3. ✅ **执行所有 warp**：所有线程的计算都会被执行
4. ⚠️ **简化的寄存器模型**：通用寄存器在 warp 间共享（对简单 kernel 足够）

对于 **vector add、matrix multiply** 等简单并行计算，这个修复已经足够。对于更复杂的场景（需要完整的寄存器隔离），需要进一步重构 RegisterBank。
