# PTX VM 文档中心

欢迎来到 PTX VM (Parallel Thread Execution Virtual Machine) 文档中心。

**作者**: Han-Zhenzhong, TongyiLingma, GitHub Copilot  
**最后更新**: 2025-10-29

## 📚 文档分类

### 🚀 [用户文档 (user_docs/)](./user_docs/)
**适合**: 使用 PTX VM 的开发者和用户

包含内容：
- 用户指南和快速入门
- 命令行界面使用说明
- PTX 代码示例和最佳实践
- API 文档
- 日志系统文档
- 功能特性说明

👉 [查看用户文档 README](./user_docs/README.md)

### 🔧 [开发文档 (dev_docs/)](./dev_docs/)
**适合**: 参与 PTX VM 开发的贡献者

包含内容：
- 开发者指南和项目架构
- 实现总结和技术分析
- 功能实现详解
- 性能优化方案
- 开发计划和改进建议

👉 [查看开发文档 README](./dev_docs/README.md)

### 📖 [规范文档 (spec_docs/)](./spec_docs/)
**适合**: 学习 PTX 基础知识和规范的读者

包含内容：
- PTX 核心概念
- SIMT 执行模型
- Warp 调度和线程分歧
- 重汇聚机制
- 谓词执行详解

👉 [查看规范文档 README](./spec_docs/README.md)

### 📋 [API 文档 (docs/)](./docs/)
**适合**: 需要详细 API 参考的开发者

包含内容：
- 完整的 API 文档
- 项目架构文档
- 技术参考资料

## 🎯 快速导航

### 我想...

#### 开始使用 PTX VM
1. 阅读 [用户指南](./user_docs/user_guide.md)
2. 查看 [快速参考](./user_docs/quick_reference.md)
3. 学习 [PTX 使用示例](./user_docs/correct_ptx_usage_examples.md)

#### 学习 PTX 基础知识
1. 了解 [CUDA C 和 PTX 的关系](./spec_docs/how_CudaC_and_PTX_called_by_HostC.md)
2. 学习 [Warp 调度机制](./spec_docs/warp_scheduler.md)
3. 理解 [线程分歧处理](./spec_docs/divergence_handling.md)

#### 参与项目开发
1. 阅读 [开发者指南](./dev_docs/developer_guide.md)
2. 查看 [项目架构分析](./dev_docs/comprehensive_implementation_analysis.md)
3. 了解 [性能测试方法](./dev_docs/performance_testing.md)

#### 查找 API 文档
1. 查看 [API 文档总览](./user_docs/api_documentation.md)
2. 浏览 [详细 API 文档](./user_docs/api_docs/)

## 📊 文档结构图

```
ptx-vm/
├── user_docs/          # 用户文档
│   ├── README.md
│   ├── user_guide.md
│   ├── quick_reference.md
│   ├── api_documentation.md
│   └── ...
├── dev_docs/           # 开发文档
│   ├── README.md
│   ├── developer_guide.md
│   ├── performance_testing.md
│   └── ...
├── spec_docs/          # 规范文档
│   ├── README.md
│   ├── warp_scheduler.md
│   ├── divergence_handling.md
│   └── ...
└── docs/               # API 和技术文档
    ├── api_documentation.md
    └── ...
```

## 🔗 外部资源

- [NVIDIA PTX ISA 官方文档](https://docs.nvidia.com/cuda/parallel-thread-execution/)
- [CUDA C 编程指南](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [项目 GitHub](https://github.com/Han-Zhenzhong/ptx-vm)

## 📝 贡献文档

欢迎贡献文档！请参考：
- [开发者指南](./dev_docs/developer_guide.md)
- 文档应放在对应的目录：
  - 用户使用相关 → `user_docs/`
  - 开发实现相关 → `dev_docs/`
  - PTX 规范知识 → `spec_docs/`

## ❓ 需要帮助？

如果找不到需要的文档，请：
1. 使用文档搜索功能
2. 查看各目录的 README
3. 提交 Issue 请求文档

---

**最后更新**: 2025-10-29
