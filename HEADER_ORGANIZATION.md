# PTX-VM 头文件组织结构建议

## 当前状态分析

### 项目根目录 `include/` (公共API)
当前位于根目录的头文件：
- `vm.hpp` - 虚拟机主接口 ✓
- `host_api.hpp` - Host API接口 ✓
- `cli_interface.hpp` - 命令行接口 ✓
- `cuda_binary_loader.hpp` - CUDA二进制加载器 ✓
- `debugger.hpp` - 调试器接口 ✓
- `performance_counters.hpp` - 性能计数器 ✓
- `instruction_types.hpp` - 指令类型定义 ✓

### 模块内部 `src/` 目录
当前位于各模块目录的头文件：
- `src/memory/memory.hpp` - 内存子系统
- `src/memory/memory_optimizer.hpp` - 内存优化器
- `src/registers/register_bank.hpp` - 寄存器组
- `src/execution/executor.hpp` - 执行器
- `src/execution/warp_scheduler.hpp` - Warp调度器
- `src/execution/predicate_handler.hpp` - 谓词处理器
- `src/execution/reconvergence_mechanism.hpp` - 重聚机制
- `src/parser/parser.hpp` - PTX解析器
- `src/decoder/decoder.hpp` - 指令解码器
- `src/optimizer/register_allocator.hpp` - 寄存器分配器
- `src/optimizer/instruction_scheduler.hpp` - 指令调度器

---

## 重构建议

### 原则

**应该放在 `include/` 的头文件：**
1. **公共API** - 外部用户直接使用的接口
2. **稳定接口** - 不经常变更的接口
3. **跨模块依赖** - 被多个模块引用的核心类型
4. **导出符号** - 需要导出给库用户的类和函数

**应该放在 `src/*/` 的头文件：**
1. **内部实现** - 模块内部实现细节
2. **频繁变更** - 实现层面经常修改的接口
3. **模块专有** - 只在特定模块内使用的类型
4. **实现依赖** - 与具体实现强耦合的接口

---

## 推荐的目录结构

### 保持在 `include/` 目录（公共API层）

```
include/
├── vm.hpp                          # ✓ 主要公共API
├── host_api.hpp                    # ✓ Host运行时API
├── cli_interface.hpp               # ✓ 命令行接口
├── cuda_binary_loader.hpp          # ✓ 二进制加载器
├── debugger.hpp                    # ✓ 调试器接口
├── performance_counters.hpp        # ✓ 性能计数器
├── instruction_types.hpp           # ✓ 指令类型（基础类型）
└── ptx_vm_api.h                    # 新增：C接口（如果需要）
```

**理由：**
- 这些是用户直接使用的接口
- 提供稳定的ABI
- 已经使用Pimpl模式隐藏实现细节
- 是项目对外导出的主要接口

### 移到 `include/` 目录（核心抽象层）

建议将以下头文件移至 `include/` 的子目录：

```
include/
├── core/
│   ├── memory.hpp                  # 从 src/memory/memory.hpp 移动
│   ├── register_bank.hpp           # 从 src/registers/register_bank.hpp 移动
│   └── executor.hpp                # 从 src/execution/executor.hpp 移动
└── types/
    └── instruction_types.hpp       # 已存在，保持不变
```

**理由：**
- `memory.hpp` - 被多个公共API依赖（vm.hpp, debugger.hpp, executor.hpp）
- `register_bank.hpp` - 被多个公共API依赖（vm.hpp, debugger.hpp）
- `executor.hpp` - 被公共API依赖（vm.hpp, debugger.hpp）
- 这些是核心抽象，虽然有Impl，但接口相对稳定
- 移到include后更清晰地表明这些是核心组件

### 保持在 `src/` 目录（内部实现层）

```
src/
├── memory/
│   ├── memory.cpp
│   └── memory_optimizer.hpp        # ✓ 内部优化实现
├── registers/
│   └── register_bank.cpp
├── execution/
│   ├── executor.cpp
│   ├── warp_scheduler.hpp          # ✓ 内部调度实现
│   ├── predicate_handler.hpp       # ✓ 内部谓词处理
│   └── reconvergence_mechanism.hpp # ✓ 内部重聚机制
├── parser/
│   ├── parser.hpp                  # ✓ PTX解析器
│   └── parser.cpp
├── decoder/
│   ├── decoder.hpp                 # ✓ 指令解码器
│   └── decoder.cpp
├── optimizer/
│   ├── register_allocator.hpp      # ✓ 寄存器分配器
│   ├── register_allocator.cpp
│   ├── instruction_scheduler.hpp   # ✓ 指令调度器
│   └── instruction_scheduler.cpp
└── core/
    ├── vm.cpp
    └── vm_profiler.cpp
```

**理由：**
- 这些是实现细节，不需要对外暴露
- 频繁修改，不影响公共API
- 只在模块内部或相关实现中使用
- 保持在src/使编译依赖更清晰

---

## 具体重构步骤

### 第一阶段：创建新的include结构（不影响现有代码）

```bash
# 创建新的子目录
mkdir -p include/core
mkdir -p include/types

# 移动核心头文件（保留src中的副本，先测试）
cp src/memory/memory.hpp include/core/
cp src/registers/register_bank.hpp include/core/
cp src/execution/executor.hpp include/core/
```

### 第二阶段：更新include路径

更新引用这些头文件的代码：

```cpp
// 旧的引用方式
#include "memory/memory.hpp"
#include "registers/register_bank.hpp"
#include "execution/executor.hpp"

// 新的引用方式
#include "core/memory.hpp"
#include "core/register_bank.hpp"
#include "core/executor.hpp"
```

需要更新的文件：
- `include/vm.hpp`
- `include/debugger.hpp`
- `src/core/vm.cpp`
- `src/debugger/debugger.cpp`
- 所有测试文件

### 第三阶段：删除旧文件

确认所有引用都已更新后：
```bash
rm src/memory/memory.hpp
rm src/registers/register_bank.hpp
rm src/execution/executor.hpp
```

### 第四阶段：更新CMakeLists.txt

```cmake
# 更新include目录
target_include_directories(ptx_vm
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# 安装公共头文件
install(DIRECTORY include/
    DESTINATION include
    FILES_MATCHING PATTERN "*.hpp" PATTERN "*.h"
)
```

---

## 使用场景对比

### 外部用户视角（使用库）

```cpp
// 只需要include公共API
#include <ptx_vm/vm.hpp>
#include <ptx_vm/host_api.hpp>
#include <ptx_vm/debugger.hpp>

// 如果需要访问核心类型（已通过公共API暴露）
#include <ptx_vm/core/memory.hpp>
#include <ptx_vm/core/register_bank.hpp>
```

### 内部开发者视角（开发库）

```cpp
// 可以访问所有头文件
#include "vm.hpp"                           // 公共API
#include "core/memory.hpp"                  // 核心抽象
#include "execution/warp_scheduler.hpp"     // 内部实现
#include "parser/parser.hpp"                // 内部实现
```

---

## 依赖关系图

```
层级划分：

┌─────────────────────────────────────────────┐
│  公共API层 (include/)                        │
│  - vm.hpp, host_api.hpp, debugger.hpp       │
│  - 使用Pimpl，依赖核心抽象层                  │
└──────────────────┬──────────────────────────┘
                   │ depends on
┌──────────────────▼──────────────────────────┐
│  核心抽象层 (include/core/)                  │
│  - memory.hpp, register_bank.hpp             │
│  - executor.hpp                              │
│  - 使用Pimpl，依赖基础类型                    │
└──────────────────┬──────────────────────────┘
                   │ depends on
┌──────────────────▼──────────────────────────┐
│  基础类型层 (include/types/)                 │
│  - instruction_types.hpp                     │
│  - 纯数据结构和枚举，无依赖                   │
└─────────────────────────────────────────────┘

实现层 (src/*/)
- 所有.cpp文件
- 内部实现头文件（warp_scheduler.hpp等）
- 依赖所有上层接口
```

---

## 优势分析

### 这种组织的好处：

1. **清晰的API边界**
   - `include/` - 公共接口，保证稳定性
   - `include/core/` - 核心抽象，相对稳定
   - `src/*/` - 实现细节，可以频繁修改

2. **更好的编译隔离**
   - 实现变更不影响公共API
   - 减少不必要的重新编译

3. **更容易的版本管理**
   - 公共API变更需要仔细审查
   - 实现层变更可以更自由

4. **更简单的安装和使用**
   - 用户只需要 `include/` 目录
   - 实现细节不会泄露

5. **更好的文档组织**
   - API文档只需关注 `include/`
   - 内部文档可以分开维护

---

## 兼容性考虑

### 向后兼容过渡方案：

在移动文件后，可以在旧位置创建转发头文件：

```cpp
// src/memory/memory.hpp (转发头文件)
#ifndef PTX_VM_MEMORY_HPP_DEPRECATED
#define PTX_VM_MEMORY_HPP_DEPRECATED

#warning "memory/memory.hpp is deprecated, use core/memory.hpp instead"
#include "core/memory.hpp"

#endif
```

这样旧代码仍能编译，但会收到警告，鼓励升级。

---

## 总结

### 立即可做（无破坏性）：

1. 在 `include/` 下创建 `core/` 子目录
2. 复制核心头文件到新位置
3. 逐步更新引用
4. 保留旧文件作为转发

### 长期目标（可选）：

1. 所有公共API移至 `include/`
2. 所有实现细节保持在 `src/`
3. 建立清晰的分层架构
4. 完善文档和示例

### 当前推荐的最小改动：

**保持现状**，因为：
- 已经使用了Pimpl模式，实现已经隐藏
- 头文件组织虽然可以改进，但不影响使用
- 重构成本需要权衡收益

**如果要改进**，优先考虑：
- 将 `instruction_types.hpp` 移至 `include/types/`
- 添加总的 `ptx_vm.hpp` 作为包含所有公共API的便捷头文件
