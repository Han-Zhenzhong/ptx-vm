# 文档更新总结 - 日志系统使用说明

**作者**: Han-Zhenzhong, GitHub Copilot  
**日期**: 2025-10-30

## 概述

本文档记录了为日志系统功能更新的所有用户文档，确保用户能够在各处文档中了解如何使用日志级别控制功能。

## 更新的文档列表

### 1. README.md（主文档）
**位置**: `d:\open-source\ptx-vm\README.md`  
**更新状态**: ✅ 已包含日志系统文档

**更新内容**:
- 在 "Usage" 部分添加了 "Log Level Control" 小节
- 包含命令行和交互式模式的日志级别控制示例
- 添加了指向详细日志系统文档的链接

**示例代码**:
```bash
# 命令行使用
./ptx_vm --log-level debug program.ptx
./ptx_vm -l info program.ptx

# 交互式模式
> loglevel debug
> loglevel warning
```

### 2. user_docs/user_guide.md（用户指南）
**位置**: `d:\open-source\ptx-vm\user_docs\user_guide.md`  
**更新状态**: ✅ 已完成更新

**更新内容**:
- 在 "Basic Usage" 部分添加了日志级别控制说明
- 包含 `--log-level`/`-l` 命令行选项的详细说明
- 提供了四个日志级别的用途说明
- 添加了使用示例

**新增章节**:
```markdown
#### Log Level Control

Control the verbosity of output using the `--log-level` or `-l` option:

- **debug**: Detailed debugging information (most verbose)
- **info**: General information (default)
- **warning**: Warnings and errors only
- **error**: Errors only (least verbose)
```

### 3. user_docs/quick_reference.md（快速参考）
**位置**: `d:\open-source\ptx-vm\user_docs\quick_reference.md`  
**更新状态**: ✅ 已完成更新

**更新内容**:
- 添加了 "快速开始" 部分
- 包含基本命令和日志命令参考
- 列出所有四个日志级别

**新增内容**:
```markdown
### 快速开始

#### 基本命令
```bash
./ptx_vm program.ptx                    # 执行 PTX 程序
./ptx_vm --log-level debug program.ptx  # 调试模式
./ptx_vm                                # 交互模式
```

#### 日志命令
- `debug` - 详细调试信息
- `info` - 常规信息（默认）
- `warning` - 警告和错误
- `error` - 仅错误
```

### 4. user_docs/USER_GUIDE_CN.md（中文用户指南）
**位置**: `d:\open-source\ptx-vm\user_docs\USER_GUIDE_CN.md`  
**更新状态**: ✅ 已完成更新

**更新内容**:
- 在 "基本使用方法" 部分添加了日志级别控制说明（中文）
- 在 "交互式命令" 部分添加了 "日志控制" 小节
- 包含完整的命令行选项和交互式命令说明
- 提供了中文的日志级别说明

**新增章节**:
1. "基本使用方法" → "日志级别控制"
2. "交互式命令" → "6. 日志控制"

**示例**:
```markdown
#### 日志级别控制

使用 `--log-level` 或 `-l` 选项控制输出详细程度：

- **debug**: 详细的调试信息（最详细）
- **info**: 一般信息（默认）
- **warning**: 警告和错误信息
- **error**: 仅错误信息（最精简）
```

### 5. user_docs/api_documentation.md（API 文档）
**位置**: `d:\open-source\ptx-vm\user_docs\api_documentation.md`  
**更新状态**: ✅ 已完成更新

**更新内容**:
- 更新了文档日期为 2025-10-30
- 在 "Overview" 部分后添加了 "Quick Start" 大节
- 包含基本使用示例和日志系统集成说明
- 提供了 Logger API 的编程接口使用示例

**新增章节**:
```markdown
## Quick Start

### Basic Usage Example
（包含完整的 C++ 代码示例，展示如何在代码中使用 Logger）

### Logging System Integration
（包含 Logger::setLogLevel()、setShowTimestamp()、setColorOutput() 等 API 说明）
```

### 6. user_docs/logging_system.md（日志系统专用文档）
**位置**: `d:\open-source\ptx-vm\user_docs\logging_system.md`  
**更新状态**: ✅ 已存在（之前创建）

**内容**:
- 完整的日志系统设计说明
- 所有日志级别的详细说明
- 命令行和交互式使用方法
- C++ API 编程接口
- 配置选项说明
- 最佳实践建议

### 7. user_docs/README.md（用户文档索引）
**位置**: `d:\open-source\ptx-vm\user_docs\README.md`  
**更新状态**: ✅ 已包含日志系统文档

**内容**:
- 在 "功能特性" 部分列出了 logging_system.md
- 说明：日志系统文档（级别控制、配置、最佳实践）

### 8. DOCS_INDEX.md（文档中心）
**位置**: `d:\open-source\ptx-vm\DOCS_INDEX.md`  
**更新状态**: ✅ 已包含日志系统文档

**内容**:
- 在用户文档分类中提到了日志系统文档

## 文档更新原则

### 1. 内容完整性
- ✅ 所有主要用户文档都包含了日志系统使用说明
- ✅ 中英文文档都已更新
- ✅ 提供了命令行和交互式两种使用方式

### 2. 一致性
- ✅ 所有文档中的命令格式保持一致
- ✅ 日志级别名称统一：debug, info, warning, error
- ✅ 选项格式统一：--log-level 或 -l

### 3. 易用性
- ✅ 在快速入门和基本使用部分添加了日志控制说明
- ✅ 提供了简单易懂的示例
- ✅ 包含了指向详细文档的链接

## 文档覆盖范围

### 已更新的文档类型
1. **主项目 README** - 让新用户第一时间了解日志功能
2. **用户指南** - 详细的使用说明（中英文）
3. **快速参考** - 便于查询的简明指南
4. **API 文档** - 编程接口说明
5. **专用文档** - 日志系统详细文档

### 文档更新位置
- 项目根目录: README.md
- 用户文档目录: user_docs/
  - user_guide.md
  - quick_reference.md
  - USER_GUIDE_CN.md
  - api_documentation.md
  - logging_system.md（专用文档）
  - README.md（索引）

## 日志系统使用总结

### 命令行方式
```bash
# 长选项
./ptx_vm --log-level debug program.ptx
./ptx_vm --log-level info program.ptx
./ptx_vm --log-level warning program.ptx
./ptx_vm --log-level error program.ptx

# 短选项
./ptx_vm -l debug program.ptx
./ptx_vm -l info program.ptx
./ptx_vm -l warning program.ptx
./ptx_vm -l error program.ptx

# 查看帮助
./ptx_vm --help
./ptx_vm -h
```

### 交互式方式
```bash
# 进入交互模式
./ptx_vm

# 在交互模式中切换日志级别
> loglevel debug
> loglevel info
> loglevel warning
> loglevel error
```

### C++ API 方式
```cpp
#include "logger.hpp"

// 设置日志级别
Logger::setLogLevel(LogLevel::DEBUG);
Logger::setLogLevel(LogLevel::INFO);    // 默认
Logger::setLogLevel(LogLevel::WARNING);
Logger::setLogLevel(LogLevel::ERROR);

// 使用日志
Logger::debug("详细调试信息");
Logger::info("一般信息");
Logger::warning("警告信息");
Logger::error("错误信息");

// 其他配置
Logger::setShowTimestamp(true);   // 显示时间戳
Logger::setColorOutput(true);     // 彩色输出
```

## 日志级别说明

| 级别 | 英文名称 | 用途 | 输出内容 |
|------|---------|------|----------|
| 1 | DEBUG | 详细调试 | 所有日志（最详细）|
| 2 | INFO | 一般信息 | INFO、WARNING、ERROR（默认）|
| 3 | WARNING | 警告 | WARNING、ERROR |
| 4 | ERROR | 错误 | 仅 ERROR（最精简）|

## 未来改进建议

1. **示例代码补充**: 在 examples/ 目录中添加使用 Logger 的示例代码
2. **博客文章**: 可以在 blog/ 目录中添加关于日志系统使用的文章
3. **视频教程**: 制作视频演示日志级别控制的效果
4. **FAQ 文档**: 添加常见问题解答

## 验证清单

- [x] README.md 已包含日志系统说明
- [x] user_docs/user_guide.md 已更新
- [x] user_docs/quick_reference.md 已更新
- [x] user_docs/USER_GUIDE_CN.md 已更新
- [x] user_docs/api_documentation.md 已更新
- [x] user_docs/logging_system.md 已存在
- [x] user_docs/README.md 已列出日志文档
- [x] DOCS_INDEX.md 已包含日志文档引用
- [x] 中英文文档都已覆盖
- [x] 命令行和交互式使用方式都已说明
- [x] C++ API 使用方式已说明

## 结论

所有主要用户文档已完成日志系统使用说明的更新。用户现在可以在以下文档中找到如何使用日志级别控制功能：

1. **快速入门**: README.md 和 quick_reference.md
2. **详细使用**: user_guide.md 和 USER_GUIDE_CN.md
3. **编程参考**: api_documentation.md
4. **深入学习**: logging_system.md

文档更新确保了日志系统功能对用户是透明和易用的。

---

**文档更新完成时间**: 2025-10-30
**审核状态**: 已完成
