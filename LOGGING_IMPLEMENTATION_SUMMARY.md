# Logging System Implementation - Change Summary

**Date**: 2025-10-30  
**Author**: Han-Zhenzhong, GitHub Copilot

## 概述 (Overview)

实现了可按级别打印的日志系统，默认级别为 INFO。系统支持 DEBUG、INFO、WARNING、ERROR 四个日志级别，可通过命令行参数或交互式命令动态调整。

Implemented a leveled logging system with default log level INFO. The system supports four log levels: DEBUG, INFO, WARNING, and ERROR, which can be adjusted via command-line options or interactive commands.

## 新增文件 (New Files)

### 核心实现 (Core Implementation)
- `include/logger.hpp` - Logger 类头文件，定义日志级别和公共 API
- `src/logger/logger.cpp` - Logger 实现，包含线程安全、颜色输出、时间戳支持
- `src/logger/CMakeLists.txt` - Logger 模块构建配置

### 文档 (Documentation)
- `user_docs/logging_system.md` - 用户文档：日志系统使用指南
- `dev_docs/logging_system_implementation.md` - 开发者文档：实现总结

## 修改文件 (Modified Files)

### 构建系统 (Build System)
- `CMakeLists.txt` - 添加 logger 子目录，链接到主程序
- `src/debugger/CMakeLists.txt` - 链接 logger 库
- `src/host/CMakeLists.txt` - 链接 logger 库

### 代码集成 (Code Integration)
- `src/main.cpp` - 错误消息使用 Logger::error()
- `src/debugger/debugger.cpp` - 错误消息使用 Logger::error()
- `src/host/cli_interface.cpp` - 添加日志级别支持：
  - 命令行选项: `--log-level` / `-l`
  - 交互式命令: `loglevel [level]`
  - 帮助信息更新

### 文档更新 (Documentation Updates)
- `README.md` - 添加日志系统使用说明和快速参考
- `DOCS_INDEX.md` - 添加日志系统到文档索引
- `user_docs/README.md` - 添加 logging_system.md 到文件列表
- `dev_docs/README.md` - 添加 logging_system_implementation.md
- `dev_docs/developer_guide.md` - 添加日志使用指南和最佳实践

## 功能特性 (Features)

### 日志级别 (Log Levels)
1. **DEBUG** - 详细调试信息
2. **INFO** (默认) - 一般信息消息
3. **WARNING** - 警告消息
4. **ERROR** - 错误消息

### 配置选项 (Configuration)
- 命令行设置: `ptx_vm --log-level debug program.ptx`
- 交互式设置: `loglevel debug`
- 查看当前级别: `loglevel`
- 时间戳开关: `Logger::setShowTimestamp(bool)`
- 颜色输出开关: `Logger::setColorOutput(bool)`

### API 使用 (API Usage)
```cpp
Logger::debug("详细调试信息");
Logger::info("正常操作消息");
Logger::warning("警告消息");
Logger::error("错误消息");
```

## 设计特点 (Design Features)

1. **默认级别 INFO** - 平衡详细度和可用性
2. **静态方法** - 简化调用，无需实例化
3. **线程安全** - 使用 mutex 保护，支持多线程
4. **ANSI 颜色** - 增强可读性（可选）
5. **可选时间戳** - 支持性能分析场景
6. **最小集成** - 仅更新错误消息，保留用户界面输出

## 使用示例 (Usage Examples)

### 命令行 (Command Line)
```bash
# 启用所有日志
ptx_vm --log-level debug program.ptx

# 默认级别 (INFO)
ptx_vm program.ptx

# 仅错误
ptx_vm --log-level error program.ptx
```

### 交互式 (Interactive)
```
ptx-vm> loglevel debug      # 设置为 DEBUG
ptx-vm> loglevel            # 查看当前级别
ptx-vm> help loglevel       # 查看帮助
```

## 测试建议 (Testing Recommendations)

1. 编译项目确认无编译错误
2. 测试不同日志级别的输出
3. 验证命令行参数解析
4. 测试交互式命令
5. 检查颜色输出
6. 验证线程安全性

## 后续改进 (Future Improvements)

- 日志文件输出
- 日志轮转
- 按模块设置级别
- 结构化日志格式
- 性能分析集成

## 文档链接 (Documentation Links)

- 用户指南: `user_docs/logging_system.md`
- 实现总结: `dev_docs/logging_system_implementation.md`
- 开发者指南: `dev_docs/developer_guide.md`
- 主文档索引: `DOCS_INDEX.md`
