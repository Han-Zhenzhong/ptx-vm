# libptxrt 开发者快速参考

## 快速开始

### 构建库

**Linux/macOS:**
```bash
./build.sh
```

**Windows:**
```cmd
build.bat
```

**手动构建:**
```bash
mkdir build && cd build
cmake ..
make
```

## API 函数分类

### 1. 内存管理
```c
cudaError_t cudaMalloc(void** devPtr, size_t size);
cudaError_t cudaFree(void* devPtr);
cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, enum cudaMemcpyKind kind);
cudaError_t cudaMemset(void* devPtr, int value, size_t count);
```

### 2. 内核启动
```c
// 直接启动
cudaError_t cudaLaunchKernel(const void* func, dim3 gridDim, dim3 blockDim,
                             void** args, size_t sharedMem, cudaStream_t stream);

// <<<>>> 语法支持
cudaError_t cudaConfigureCall(dim3 gridDim, dim3 blockDim, size_t sharedMem, cudaStream_t stream);
cudaError_t cudaSetupArgument(const void* arg, size_t size, size_t offset);
cudaError_t cudaLaunch(const void* func);
```

### 3. 注册函数（编译器生成）
```c
void** __cudaRegisterFatBinary(void* fatCubin);
void __cudaUnregisterFatBinary(void** fatCubinHandle);
void __cudaRegisterFunction(void** fatCubinHandle, const char* hostFun, ...);
```

### 4. 同步
```c
cudaError_t cudaDeviceSynchronize(void);
cudaError_t cudaStreamSynchronize(cudaStream_t stream);
cudaError_t cudaEventSynchronize(cudaEvent_t event);
```

## 实现优先级

### 阶段 1: 基础功能（必需）
1. ✅ `__cudaRegisterFatBinary` - 提取 PTX
2. ✅ `__cudaRegisterFunction` - 注册内核
3. ✅ `cudaMalloc` / `cudaFree` - 内存管理
4. ✅ `cudaMemcpy` - 数据传输
5. ✅ `cudaLaunchKernel` - 启动内核

### 阶段 2: 完善功能
6. ✅ `cudaConfigureCall` / `cudaSetupArgument` / `cudaLaunch` - <<<>>> 支持
7. ✅ `cudaDeviceSynchronize` - 同步
8. ✅ 错误处理机制

### 阶段 3: 高级功能
9. ⏸️ 流管理
10. ⏸️ 事件管理
11. ⏸️ 多设备支持

## 内部数据结构

### RuntimeState（全局单例）
```cpp
class RuntimeState {
    std::map<void**, FatBinaryInfo> fat_binaries;     // Fat binary 信息
    std::map<void*, DeviceMemory> device_memory;      // 设备内存映射
    LaunchConfig* current_launch_config;              // 当前启动配置
    cudaError_t last_error;                           // 最后错误
};
```

### FatBinaryInfo
```cpp
struct FatBinaryInfo {
    std::vector<PTXCode> ptx_codes;      // PTX 代码列表
    std::vector<KernelInfo> kernels;     // 注册的内核
};
```

### KernelInfo
```cpp
struct KernelInfo {
    std::string kernel_name;    // 内核名（PTX 中）
    const void* host_func;      // Host 函数指针
    PTXCode* ptx_code;          // 关联的 PTX
};
```

## 工作流程

### 程序启动时
```
1. 编译器生成的初始化代码调用
   __cudaRegisterFatBinary(fatbin_data)
   └─> 提取 PTX 代码
   └─> 存储到 RuntimeState

2. 对每个 kernel 调用
   __cudaRegisterFunction(handle, host_ptr, kernel_name)
   └─> 建立 host_ptr -> kernel_name 映射
```

### 内核启动时
```
方式1: 直接调用
cudaLaunchKernel(func, grid, block, args, ...)
└─> 查找 KernelInfo
└─> 获取 PTX 代码
└─> 调用 PTX VM 执行

方式2: <<<>>> 语法
kernel<<<grid, block>>>(args...)
编译器转换为:
├─> cudaConfigureCall(grid, block, ...)
├─> cudaSetupArgument(&arg1, ...)
├─> cudaSetupArgument(&arg2, ...)
└─> cudaLaunch(func)
    └─> 使用保存的配置启动
```

## PTX VM 接口（待实现）

```cpp
// 预期的 PTX VM 接口
namespace ptx_vm {
    bool execute(
        const char* ptx_code,       // PTX 源代码
        const char* kernel_name,    // 内核名称
        dim3 grid_dim,              // Grid 维度
        dim3 block_dim,             // Block 维度
        void** args,                // 参数数组
        size_t shared_mem           // 共享内存大小
    );
}
```

## Fat Binary 格式

### 结构
```
.nv_fatbin section:
├─ Header (magic, version)
├─ PTX entries
│  ├─ PTX for sm_60
│  ├─ PTX for sm_61
│  └─ PTX for sm_70
└─ CUBIN entries (忽略)
```

### 提取方法
```cpp
// 伪代码
void** __cudaRegisterFatBinary(void* fatCubin) {
    auto wrapper = (FatBinWrapper*)fatCubin;
    
    // 1. 验证 magic number
    // 2. 遍历所有条目
    // 3. 提取 PTX 文本
    // 4. 存储到 FatBinaryInfo
    
    return handle;
}
```

## 编译示例

### 使用 Clang
```bash
clang++ vector_add.cu \
  --cuda-path=/usr/local/cuda \
  --cuda-gpu-arch=sm_61 \
  -I/path/to/cuda_runtime \
  -L/path/to/build \
  -lptxrt \
  -o vector_add
```

### 验证 PTX 生成
```bash
clang++ vector_add.cu \
  --cuda-device-only \
  --cuda-gpu-arch=sm_61 \
  -S -o vector_add.ptx

cat vector_add.ptx
```

## 调试技巧

### 1. 打印 Fat Binary 信息
```cpp
void** __cudaRegisterFatBinary(void* fatCubin) {
    printf("Fat binary registered at %p\n", fatCubin);
    // ... 解析并打印内容
}
```

### 2. 跟踪内核注册
```cpp
void __cudaRegisterFunction(..., const char* deviceName, ...) {
    printf("Kernel registered: %s at %p\n", deviceName, hostFun);
}
```

### 3. 监控内存操作
```cpp
cudaError_t cudaMalloc(void** devPtr, size_t size) {
    void* ptr = malloc(size);
    printf("cudaMalloc: %zu bytes at %p\n", size, ptr);
    *devPtr = ptr;
    return cudaSuccess;
}
```

## 常见问题

### Q: 如何确定使用哪个 SM 版本？
A: PTX 6.1 对应 sm_60 或 sm_61。推荐使用 sm_61。

### Q: 如何测试库是否工作？
A: 使用 examples/simple_add.cu，添加 printf 调试输出。

### Q: 内存管理如何实现？
A: 用 malloc/free 模拟，用 map 跟踪分配。

### Q: 如何连接 PTX VM？
A: 在 cudaLaunchKernel 中调用 PTX VM 的执行接口。

## 下一步

1. 实现 Fat Binary 解析器
2. 实现内核注册表
3. 实现简单的内存管理
4. 连接 PTX VM 执行接口
5. 测试 simple_add 示例

## 参考资料

- CUDA Runtime API: https://docs.nvidia.com/cuda/cuda-runtime-api/
- PTX ISA: https://docs.nvidia.com/cuda/parallel-thread-execution/
- Fat Binary 格式: 使用 `cuobjdump` 工具分析
