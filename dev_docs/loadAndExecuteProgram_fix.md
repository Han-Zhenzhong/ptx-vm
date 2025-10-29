# loadAndExecuteProgram 设计问题修复

## 🐛 问题描述

### 当前调用栈

```
CLI::loadCommand() / CLI::loadProgram()
    ↓
PTXVM::loadAndExecuteProgram(filename)
    ↓
PTXParser::parseFile()
    ↓
PTXExecutor::initialize(program)
    ↓
PTXExecutor::execute()  ← ❌ 不应该在 load 时执行！
```

### 问题代码

**文件**：`src/core/vm.cpp` 第 205-227 行

```cpp
bool PTXVM::loadAndExecuteProgram(const std::string& filename) {
    // Create a parser and parse the file
    PTXParser parser;
    if (!parser.parseFile(filename)) {
        std::cerr << "Failed to parse PTX file: " << filename << std::endl;
        std::cerr << "Error: " << parser.getErrorMessage() << std::endl;
        return false;
    }
    
    // Get the complete PTX program
    const PTXProgram& program = parser.getProgram();
    
    // Initialize executor with the complete PTX program (not just instructions)
    if (!pImpl->m_executor->initialize(program)) {
        std::cerr << "Failed to initialize executor with PTX program" << std::endl;
        return false;
    }
    
    std::cout << "Successfully loaded PTX program from: " << filename << std::endl;
    
    // ❌ 问题：load 操作不应该执行程序！
    return pImpl->m_executor->execute();
}
```

---

## ❌ 为什么这是错误的

### 1. 破坏了正常的 CUDA 工作流

CUDA/PTX 的标准工作流是：

```cpp
// 1. 加载程序
cuModuleLoad(&module, "kernel.ptx");

// 2. 分配内存
cuMemAlloc(&d_A, size);
cuMemAlloc(&d_B, size);

// 3. 拷贝数据
cuMemcpyHtoD(d_A, h_A, size);

// 4. 获取 kernel 函数
cuModuleGetFunction(&func, module, "vecAdd");

// 5. 启动 kernel（传递设备内存地址）
cuLaunchKernel(func, ..., kernelParams, ...);
```

**当前的 `loadAndExecuteProgram` 强制在步骤 1 执行步骤 5！**

### 2. CLI 命令语义不一致

```bash
# 用户期望：
ptx-vm> load test.ptx          # 只加载程序
Program loaded successfully.

ptx-vm> alloc 32               # 分配设备内存
Allocated 32 bytes at 0x10000

ptx-vm> launch vecAdd 0x10000 0x10020 0x10040  # 执行 kernel
Kernel launched successfully.

# 实际情况：
ptx-vm> load test.ptx          # 加载 + 立即执行！
Successfully loaded PTX program from: test.ptx
[执行 vecAdd 但没有参数/内存！] ← ❌ 崩溃或错误结果
Program loaded successfully.

ptx-vm> launch vecAdd 0x10000 0x10020 0x10040
[再次执行？还是不执行？] ← ❌ 行为不明确
```

### 3. 与 `run()` 方法冲突

`src/core/vm.cpp` 中还有一个 `run()` 方法：

```cpp
bool PTXVM::run() {
    if (!pImpl->m_isProgramLoaded) {
        return false;
    }
    
    return pImpl->m_executor->execute();
}
```

这导致：
- `loadAndExecuteProgram()` 调用 `execute()` → 第 1 次执行
- 用户调用 `run` 命令 → `run()` 调用 `execute()` → 第 2 次执行
- **重复执行或状态混乱**

### 4. 测试代码也受影响

`tests/system_tests/smoke_test.cpp` 中：

```cpp
bool result = vm->loadAndExecuteProgram("examples/simple_math_example.ptx");
```

这些测试**没有设置参数、没有分配内存**，就直接执行了 kernel！
- 如果 kernel 需要参数 → 未定义行为
- 如果 kernel 访问内存 → 可能崩溃

---

## ✅ 正确的设计

### 方案 1：分离 `load` 和 `execute`（推荐）

#### 1.1 修改 `PTXVM` 接口

**文件**：`include/vm.hpp`

```cpp
class PTXVM {
public:
    // 只加载程序，不执行
    bool loadProgram(const std::string& filename);
    
    // 执行已加载的程序
    bool run();
    
    // [废弃] 仅用于向后兼容旧测试
    [[deprecated("Use loadProgram() + run() instead")]]
    bool loadAndExecuteProgram(const std::string& filename) {
        return loadProgram(filename) && run();
    }
};
```

#### 1.2 实现 `loadProgram`

**文件**：`src/core/vm.cpp`

```cpp
bool PTXVM::loadProgram(const std::string& filename) {
    // Create a parser and parse the file
    PTXParser parser;
    if (!parser.parseFile(filename)) {
        std::cerr << "Failed to parse PTX file: " << filename << std::endl;
        std::cerr << "Error: " << parser.getErrorMessage() << std::endl;
        return false;
    }
    
    // Get the complete PTX program
    const PTXProgram& program = parser.getProgram();
    
    // Initialize executor with the complete PTX program
    if (!pImpl->m_executor->initialize(program)) {
        std::cerr << "Failed to initialize executor with PTX program" << std::endl;
        return false;
    }
    
    // 标记程序已加载
    pImpl->m_isProgramLoaded = true;
    
    std::cout << "Successfully loaded PTX program from: " << filename << std::endl;
    
    // ✅ 不调用 execute()
    return true;
}

bool PTXVM::run() {
    if (!pImpl->m_isProgramLoaded) {
        std::cerr << "No program loaded. Use loadProgram() first." << std::endl;
        return false;
    }
    
    return pImpl->m_executor->execute();
}

// 向后兼容（但标记为废弃）
bool PTXVM::loadAndExecuteProgram(const std::string& filename) {
    if (!loadProgram(filename)) {
        return false;
    }
    return run();
}
```

#### 1.3 更新 CLI 接口

**文件**：`src/host/cli_interface.cpp`

```cpp
// Load command - 只加载程序
void loadCommand(const std::vector<std::string>& args) {
    if (args.empty()) {
        printError("Usage: load <filename>");
        return;
    }
    
    std::string filename = args[0];
    
    // ✅ 只加载，不执行
    if (m_vm->loadProgram(filename)) {
        m_loadedProgram = filename;
        printMessage("Program loaded successfully.");
        printMessage("Use 'launch <kernel>' to execute a kernel.");
        
        resetExecutionState();
    } else {
        printError("Failed to load program. Use 'help load' for usage.");
    }
}

// Load a program (processArguments 调用)
void loadProgram(const std::string& filename) {
    // ✅ 只加载，不执行
    if (m_vm->loadProgram(filename)) {
        m_loadedProgram = filename;
        resetExecutionState();
    }
}

// Run command - 执行整个程序（旧式，不推荐）
void runCommand(const std::vector<std::string>& args) {
    if (m_loadedProgram.empty()) {
        printError("No program loaded. Use 'load' to load a program first.");
        return;
    }
    
    printMessage("Starting program execution...");
    
    // ✅ 显式调用 run()
    bool result = m_vm->run();
    
    if (result) {
        printMessage("Program completed successfully.");
    } else {
        printError("Program execution failed.");
    }
}

// Launch command - 执行单个 kernel（新式，推荐）
void launchCommand(const std::vector<std::string>& args) {
    // ... 现有实现 ...
    // 调用 HostAPI::launchKernel()
}
```

---

## 📋 需要修改的文件清单

### 核心修改

1. **`include/vm.hpp`**
   - 添加 `bool loadProgram(const std::string& filename);`
   - 保留 `bool run();`
   - 标记 `loadAndExecuteProgram()` 为 deprecated

2. **`src/core/vm.cpp`**
   - 实现 `loadProgram()` - 只加载，不执行
   - 保持 `run()` 不变
   - 修改 `loadAndExecuteProgram()` 调用 `loadProgram() + run()`

3. **`src/host/cli_interface.cpp`**
   - 修改 `loadCommand()` 调用 `loadProgram()` 而非 `loadAndExecuteProgram()`
   - 修改 `loadProgram()` 调用 `PTXVM::loadProgram()`
   - 保持 `runCommand()` 调用 `PTXVM::run()`

### 测试修改

4. **`tests/system_tests/smoke_test.cpp`**
   ```cpp
   // 旧代码：
   bool result = vm->loadAndExecuteProgram("examples/simple_math_example.ptx");
   
   // 新代码：
   bool loaded = vm->loadProgram("examples/simple_math_example.ptx");
   ASSERT_TRUE(loaded);
   
   // 设置参数、分配内存等...
   
   bool result = vm->run();
   ASSERT_TRUE(result);
   ```

5. **`tests/system_tests/performance_test.cpp`**
   - 同上修改

6. **`src/host/host_api.cpp`**
   ```cpp
   // 第 61 行：
   // 旧代码：
   return m_vm->loadAndExecuteProgram(m_programFilename);
   
   // 新代码：
   return m_vm->loadProgram(m_programFilename);
   ```

---

## 🎯 预期行为（修复后）

### CLI 正确工作流

```bash
# 1. 加载 PTX 程序（只解析，不执行）
ptx-vm> load examples/parameter_passing_example.ptx
Successfully loaded PTX program from: examples/parameter_passing_example.ptx
Program loaded successfully.
Use 'launch <kernel>' to execute a kernel.

# 2. 分配设备内存
ptx-vm> alloc 32
Allocated 32 bytes at 0x10000

ptx-vm> alloc 32
Allocated 32 bytes at 0x10020

ptx-vm> alloc 32
Allocated 32 bytes at 0x10040

# 3. 填充数据（可选）
ptx-vm> fill 0x10000 8 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0

# 4. 启动 kernel（传递设备内存地址）
ptx-vm> launch vecAdd 0x10000 0x10020 0x10040

Parsing kernel parameters:
  [0] A (.u64): device address 0x10000
  [1] B (.u64): device address 0x10020
  [2] C (.u64): device address 0x10040

Launching kernel: vecAdd
Grid dimensions: 1 x 1 x 1
Block dimensions: 32 x 1 x 1

✓ Kernel launched successfully

# 5. 查看结果
ptx-vm> memory 0x10040 32
0x10040: ...
```

### API 正确工作流

```cpp
PTXVM vm;

// 1. 只加载
if (!vm.loadProgram("kernel.ptx")) {
    return false;
}

// 2. 分配内存
uint64_t d_A, d_B, d_C;
vm.allocateMemory(32, &d_A);
vm.allocateMemory(32, &d_B);
vm.allocateMemory(32, &d_C);

// 3. 拷贝数据
vm.copyToDevice(d_A, h_A, 32);
vm.copyToDevice(d_B, h_B, 32);

// 4. 启动 kernel
std::vector<void*> params = {&d_A, &d_B, &d_C};
vm.launchKernel("vecAdd", 1, 1, 1, 32, 1, 1, params);

// 5. 拷贝结果
vm.copyFromDevice(h_C, d_C, 32);
```

---

## ⚠️ 迁移注意事项

### 破坏性变更

如果直接修改 `loadAndExecuteProgram` 的行为，会破坏现有代码：

**影响的代码**：
- `src/host/host_api.cpp` (1 处)
- `tests/system_tests/smoke_test.cpp` (3 处)
- `tests/system_tests/performance_test.cpp` (3 处)
- 用户的外部代码（未知）

### 推荐迁移策略

1. **阶段 1**：添加新接口
   - 添加 `loadProgram()` 方法
   - 保持 `loadAndExecuteProgram()` 不变（但标记 deprecated）
   - 更新文档说明新旧接口

2. **阶段 2**：更新内部代码
   - 修改 CLI 使用 `loadProgram()`
   - 修改测试使用 `loadProgram() + run()`
   - 修改 `host_api.cpp`

3. **阶段 3**：（可选）移除旧接口
   - 确认所有代码已迁移
   - 移除 `loadAndExecuteProgram()` 或让它只是调用新接口

---

## 🔍 调用栈对比

### 修复前（错误）

```
CLI::loadCommand("test.ptx")
    ↓
PTXVM::loadAndExecuteProgram("test.ptx")
    ↓
PTXParser::parseFile()
    ↓
PTXExecutor::initialize(program)
    ↓
PTXExecutor::execute()  ← ❌ 在 load 时就执行了！
    ↓
[程序执行但没有正确的参数/内存]
    ↓
返回 CLI
```

### 修复后（正确）

```
# 步骤 1：加载
CLI::loadCommand("test.ptx")
    ↓
PTXVM::loadProgram("test.ptx")  ✅ 只加载
    ↓
PTXParser::parseFile()
    ↓
PTXExecutor::initialize(program)
    ↓
返回 CLI (不执行)

# 步骤 2：分配内存
CLI::allocCommand(32)
    ↓
HostAPI::allocateMemory(32)
    ↓
返回地址 0x10000

# 步骤 3：启动 kernel
CLI::launchCommand("vecAdd", "0x10000", "0x10020", "0x10040")
    ↓
HostAPI::launchKernel("vecAdd", params)
    ↓
cuLaunchKernel(func, ..., kernelParams, ...)
    ↓
PTXExecutor::execute()  ✅ 在 launch 时执行！
    ↓
[程序执行，有正确的参数和内存]
```

---

## ✅ 总结

### 核心问题

**`PTXVM::loadAndExecuteProgram()` 在加载时立即执行程序，破坏了 CUDA 的标准工作流。**

### 解决方案

1. 添加 `PTXVM::loadProgram()` - 只加载，不执行
2. 保持 `PTXVM::run()` - 执行已加载的程序
3. 修改 CLI 和测试使用新接口
4. 标记 `loadAndExecuteProgram()` 为 deprecated

### 优先级

**高优先级** - 这是一个设计缺陷，影响 CLI 的核心功能和用户体验。

### 相关问题

- 参数传递机制已修复（自动类型推断）
- 内存分配机制已完善（alloc/memcpy/fill）
- 缺少的是正确的**加载/执行分离**

---

**建议立即修复此问题以使 PTX VM 符合 CUDA 的标准语义。** 🎯
