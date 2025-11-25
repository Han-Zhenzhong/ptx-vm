# libptxrt 构建和测试指南

## 前提条件

1. PTX VM 已经构建在 `build/` 目录下
2. 已安装 CMake 3.10+
3. 已安装 C++ 编译器（支持 C++11）

## 构建步骤

### 方式 1: 使用快速构建脚本

```bash
cd cuda/cuda_runtime
chmod +x quick_build.sh
./quick_build.sh
```

### 方式 2: 手动构建

```bash
cd cuda/cuda_runtime
mkdir -p build && cd build
cmake ..
make
```

构建完成后，会生成：
- `libptxrt.a` - 静态库
- `libptxrt.so` - 动态库

## 测试 simple_add 示例

### 步骤 1: 提取 PTX

由于当前实现还未支持自动解析 fat binary，我们需要手动提取 PTX：

```bash
cd cuda/cuda_runtime/examples

# 使用 Clang 提取 PTX（推荐）
clang++ simple_add.cu \
  --cuda-device-only \
  --cuda-gpu-arch=sm_61 \
  -S -o simple_add.ptx

# 或者使用 NVCC
# nvcc simple_add.cu -ptx -o simple_add.ptx
```

验证 PTX 生成：
```bash
grep ".entry" simple_add.ptx
# 应该看到类似: .entry _Z9vectorAddPKfS0_Pfi(
```

### 步骤 2: 编译主机代码

```bash
# 方式 A: 使用 Clang（仅编译 host 代码）
clang++ simple_add.cu \
  --cuda-host-only \
  -I.. \
  -L../build \
  -lptxrt \
  -Wl,-rpath,../build \
  -o simple_add

# 方式 B: 使用 g++（将 .cu 当作 .cpp 处理）
g++ simple_add.cu \
  -I.. \
  -L../build \
  -lptxrt \
  -Wl,-rpath,../build \
  -o simple_add
```

### 步骤 3: 运行测试

```bash
# 设置 PTX 文件路径
export PTXRT_PTX_PATH=./simple_add.ptx

# 运行
./simple_add
```

### 预期输出

```
[libptxrt] Fat binary registered at 0x...
[libptxrt] NOTE: Please set PTX file via PTXRT_PTX_PATH environment variable
[libptxrt] Kernel registered: _Z9vectorAddPKfS0_Pfi at 0x...
[libptxrt] Launching kernel: _Z9vectorAddPKfS0_Pfi
[libptxrt] Grid: (4,1,1) Block: (256,1,1)
[libptxrt] Loading PTX: ./simple_add.ptx
[libptxrt] PTX loaded successfully
[libptxrt] Kernel launched successfully
Vector addition successful! Verified 1024 elements.
```

## 故障排查

### 问题 1: 找不到 PTX 文件
```
[libptxrt] ERROR: Failed to load PTX from ...
```

**解决方案**:
- 确保 PTX 文件存在
- 使用绝对路径: `export PTXRT_PTX_PATH=/full/path/to/simple_add.ptx`
- 检查文件权限

### 问题 2: 链接错误
```
undefined reference to `HostAPI::...`
```

**解决方案**:
- 确保 PTX VM 已经构建
- 检查 CMakeLists.txt 中的链接库配置
- 重新构建: `cd build && rm -rf * && cmake .. && make`

### 问题 3: 内核名不匹配
```
[libptxrt] ERROR: Kernel not found
```

**解决方案**:
- 查看 PTX 文件中的 `.entry` 名称: `grep ".entry" simple_add.ptx`
- 确保注册的名称与 PTX 中的匹配
- 注意 C++ name mangling（`_Z9vectorAddPKfS0_Pfi` 是 mangled 名称）

### 问题 4: 运行时找不到共享库
```
error while loading shared libraries: libptxrt.so: cannot open shared object file
```

**解决方案**:
- 使用 `-Wl,-rpath` 指定库路径
- 或设置 `LD_LIBRARY_PATH`: `export LD_LIBRARY_PATH=../build:$LD_LIBRARY_PATH`
- 或使用静态链接: `-static -lptxrt`

## 下一步

- 实现 Fat Binary 自动解析（无需手动提取 PTX）
- 支持多个内核
- 添加更多示例程序
- 优化错误处理和调试信息

## 参考文档

- `ACTION_PLAN.md` - 实现计划
- `IMPLEMENTATION_QUICKSTART.md` - 快速实现指南
- `DEVELOPER_GUIDE.md` - 开发者参考
- `KERNEL_LAUNCH_SYNTAX.md` - 内核启动语法说明
