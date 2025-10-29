# PTX 规范和基础知识文档 (PTX Specification & Fundamentals)

本目录包含 PTX (Parallel Thread Execution) 的基础知识和规范相关文档。

## 📋 目录结构

### PTX 核心概念
- **how_CudaC_and_PTX_called_by_HostC.md** - CUDA C 和 PTX 如何被 Host C 调用
- **cuda_binary_loader.md** - CUDA 二进制加载器说明

### SIMT 执行模型
- **warp_scheduler.md** - Warp 调度器详解
- **divergence_handling.md** - 分支分歧处理机制
- **reconvergence_mechanism.md** - 重汇聚机制
- **predicate_handler.md** - 谓词处理器详解

### 技术参考
- **technical_ref/** - 技术参考资料目录
- **CMakeLists.txt** - 文档构建配置

## 📚 PTX 学习路径

### 1. 基础概念
首先了解 PTX 的基本概念：
- PTX 是什么
- 为什么需要 PTX
- PTX 与 CUDA C 的关系

推荐阅读：
- `how_CudaC_and_PTX_called_by_HostC.md`
- `cuda_binary_loader.md`

### 2. SIMT 执行模型
理解 GPU 的 SIMT (Single Instruction, Multiple Threads) 执行模型：
- Warp 的概念
- 线程分歧 (Thread Divergence)
- 重汇聚 (Reconvergence)

推荐阅读：
- `warp_scheduler.md`
- `divergence_handling.md`
- `reconvergence_mechanism.md`

### 3. 高级特性
深入理解 PTX 的高级特性：
- 谓词执行
- 内存模型
- 同步机制

推荐阅读：
- `predicate_handler.md`

## 🔗 相关资源

### 官方文档
- [NVIDIA PTX ISA 文档](https://docs.nvidia.com/cuda/parallel-thread-execution/)
- [CUDA C 编程指南](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)

### 项目文档
- **用户文档**: `../user_docs/` - 如何使用 PTX VM
- **开发文档**: `../dev_docs/` - 如何开发和扩展 PTX VM

## 📖 术语表

- **PTX**: Parallel Thread Execution，并行线程执行
- **Warp**: GPU 上一组同时执行的线程（通常为 32 个）
- **SIMT**: Single Instruction, Multiple Threads，单指令多线程
- **Divergence**: 线程分歧，warp 内线程执行不同路径
- **Reconvergence**: 重汇聚，分歧的线程重新汇合
- **Predicate**: 谓词，用于条件执行的布尔值
