# 实现 libptxrt 功能的最小行动清单

## 🎯 目标
让 `simple_add.cu` 能在 PTX VM 上运行

---

## ✅ 核心任务（按顺序执行）

### 任务 1: 集成 HostAPI (30分钟)

**修改文件**: `cuda/cuda_runtime/CMakeLists.txt`

```cmake
# 在 CMakeLists.txt 中添加

# 找到 PTX VM 的头文件和库
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/include  # PTX VM 头文件
    ${CMAKE_SOURCE_DIR}/src      # 如果需要
)

# 链接 PTX VM 库
# 假设已经构建了 PTX VM，库在 build/ 目录
link_directories(
    ${CMAKE_SOURCE_DIR}/build
)

# 修改静态库链接
target_link_libraries(ptxrt_static PUBLIC
    # 添加 PTX VM 的库（名字待确认）
    # ptx_vm 或其他
)

# 修改动态库链接
target_link_libraries(ptxrt_shared PUBLIC
    # 添加 PTX VM 的库
)
```

**验证**: 编译通过，可以 `#include "host_api.hpp"`

---

### 任务 2: 实现内存管理 (1小时)

**修改文件**: `cuda/cuda_runtime/cuda_runtime.cpp`

**添加头文件**:
```cpp
#include "host_api.hpp"
#include "cuda_runtime_internal.h"
```

**修改 RuntimeState**（在 `cuda_runtime_internal.h`）:
```cpp
class RuntimeState {
public:
    HostAPI host_api;  // HostAPI 实例
    // ... 其他成员
    
    RuntimeState() {
        host_api.initialize();
        host_api.cuInit(0);
    }
};
```

**实现 3 个函数**:

```cpp
cudaError_t cudaMalloc(void** devPtr, size_t size) {
    auto& state = RuntimeState::getInstance();
    CUdeviceptr ptr;
    
    CUresult result = state.host_api.cuMemAlloc(&ptr, size);
    if (result != CUDA_SUCCESS) {
        return cudaErrorMemoryAllocation;
    }
    
    *devPtr = reinterpret_cast<void*>(ptr);
    return cudaSuccess;
}

cudaError_t cudaFree(void* devPtr) {
    auto& state = RuntimeState::getInstance();
    CUdeviceptr ptr = reinterpret_cast<CUdeviceptr>(devPtr);
    
    CUresult result = state.host_api.cuMemFree(ptr);
    return (result == CUDA_SUCCESS) ? cudaSuccess : cudaErrorUnknown;
}

cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, 
                       enum cudaMemcpyKind kind) {
    auto& state = RuntimeState::getInstance();
    CUresult result;
    
    if (kind == cudaMemcpyHostToDevice) {
        result = state.host_api.cuMemcpyHtoD(
            reinterpret_cast<CUdeviceptr>(dst), src, count);
    } else if (kind == cudaMemcpyDeviceToHost) {
        result = state.host_api.cuMemcpyDtoH(
            dst, reinterpret_cast<CUdeviceptr>(src), count);
    } else {
        return cudaErrorInvalidMemcpyDirection;
    }
    
    return (result == CUDA_SUCCESS) ? cudaSuccess : cudaErrorUnknown;
}
```

**验证**: 编译通过

---

### 任务 3: 实现内核注册 (1小时)

**修改 RuntimeState** 添加映射表:
```cpp
class RuntimeState {
public:
    std::map<const void*, KernelInfo> kernel_map;  // host指针 → 内核信息
    std::string ptx_file_path;  // PTX 文件路径
    bool program_loaded = false;
};
```

**实现 2 个函数**:

```cpp
void** __cudaRegisterFatBinary(void* fatCubin) {
    // 简化版本：不解析 fat binary
    static void* handle = malloc(8);
    
    printf("[libptxrt] Fat binary registered\n");
    printf("[libptxrt] NOTE: Please set PTX file via PTXRT_PTX_PATH env var\n");
    
    return &handle;
}

void __cudaRegisterFunction(void** fatCubinHandle,
                           const char* hostFun,
                           char* deviceFun,
                           const char* deviceName,
                           int thread_limit,
                           uint3* tid, uint3* bid,
                           dim3* bDim, dim3* gDim,
                           int* wSize) {
    auto& state = RuntimeState::getInstance();
    
    KernelInfo info;
    info.kernel_name = deviceName;
    info.host_func = (const void*)hostFun;
    
    state.kernel_map[(const void*)hostFun] = info;
    
    printf("[libptxrt] Kernel registered: %s at %p\n", 
           deviceName, hostFun);
}
```

**验证**: 编译通过

---

### 任务 4: 实现内核启动（含 <<<>>> 语法支持）(2-3小时)

**这是最关键的部分！**

**实现**:

```cpp
cudaError_t cudaLaunchKernel(const void* func, dim3 gridDim, dim3 blockDim,
                             void** args, size_t sharedMem, cudaStream_t stream) {
    auto& state = RuntimeState::getInstance();
    
    // 1. 查找内核
    auto it = state.kernel_map.find(func);
    if (it == state.kernel_map.end()) {
        fprintf(stderr, "[libptxrt] ERROR: Kernel not found\n");
        return cudaErrorInvalidValue;
    }
    
    const KernelInfo& kernel = it->second;
    printf("[libptxrt] Launching kernel: %s\n", kernel.kernel_name.c_str());
    printf("[libptxrt] Grid: (%u,%u,%u) Block: (%u,%u,%u)\n",
           gridDim.x, gridDim.y, gridDim.z,
           blockDim.x, blockDim.y, blockDim.z);
    
    // 2. 加载 PTX 程序（首次）
    if (!state.program_loaded) {
        // 从环境变量获取 PTX 路径
        const char* ptx_path = getenv("PTXRT_PTX_PATH");
        if (!ptx_path) {
            // 默认尝试 kernel_name.ptx
            state.ptx_file_path = kernel.kernel_name + ".ptx";
        } else {
            state.ptx_file_path = ptx_path;
        }
        
        printf("[libptxrt] Loading PTX: %s\n", state.ptx_file_path.c_str());
        
        if (!state.host_api.loadProgram(state.ptx_file_path)) {
            fprintf(stderr, "[libptxrt] ERROR: Failed to load PTX\n");
            return cudaErrorInvalidSource;
        }
        
        state.program_loaded = true;
    }
    
    // 3. 启动内核
    // 注意：这里需要获取 CUfunction，可能需要修改 HostAPI
    // 暂时使用 0 作为占位符
    CUfunction f = 0;  // TODO: 需要从 kernel_name 查找
    
    CUresult result = state.host_api.cuLaunchKernel(
        f,
        gridDim.x, gridDim.y, gridDim.z,
        blockDim.x, blockDim.y, blockDim.z,
        sharedMem,
        nullptr,  // stream
        args,
        nullptr   // extra
    );
    
    if (result != CUDA_SUCCESS) {
        fprintf(stderr, "[libptxrt] ERROR: Kernel launch failed\n");
        return cudaErrorLaunchFailure;
    }
    
    printf("[libptxrt] Kernel launched successfully\n");
    return cudaSuccess;
}
```

**可能需要修改 HostAPI**:
在 `include/host_api.hpp` 中添加：
```cpp
// 添加方法获取函数 handle
CUresult cuModuleGetFunction(CUfunction* hfunc, const char* name);
```

**实现 <<<>>> 语法支持**（Clang 会自动调用这些函数）:

```cpp
// 全局变量保存当前配置
static thread_local LaunchConfig* g_current_config = nullptr;

cudaError_t cudaConfigureCall(dim3 gridDim, dim3 blockDim, 
                               size_t sharedMem, cudaStream_t stream) {
    // 创建或重置配置
    if (!g_current_config) {
        g_current_config = new LaunchConfig();
    } else {
        g_current_config->args.clear();
        g_current_config->arg_sizes.clear();
    }
    
    g_current_config->grid_dim = gridDim;
    g_current_config->block_dim = blockDim;
    g_current_config->shared_mem = sharedMem;
    g_current_config->stream = stream;
    
    return cudaSuccess;
}

cudaError_t cudaSetupArgument(const void* arg, size_t size, size_t offset) {
    if (!g_current_config) {
        return cudaErrorInvalidValue;
    }
    
    // 复制参数数据
    void* arg_copy = malloc(size);
    memcpy(arg_copy, arg, size);
    
    g_current_config->args.push_back(arg_copy);
    g_current_config->arg_sizes.push_back(size);
    
    return cudaSuccess;
}

cudaError_t cudaLaunch(const void* func) {
    if (!g_current_config) {
        return cudaErrorInvalidValue;
    }
    
    // 准备参数数组
    void** args = g_current_config->args.data();
    
    // 调用 cudaLaunchKernel
    cudaError_t result = cudaLaunchKernel(
        func,
        g_current_config->grid_dim,
        g_current_config->block_dim,
        args,
        g_current_config->shared_mem,
        g_current_config->stream
    );
    
    // 清理参数副本
    for (void* arg : g_current_config->args) {
        free(arg);
    }
    g_current_config->args.clear();
    g_current_config->arg_sizes.clear();
    
    return result;
}
```

**验证**: 编译通过

---

### 任务 5: 测试 (2-3小时)

#### 5.1 提取 PTX
```bash
cd cuda/cuda_runtime/examples

# 提取 PTX
clang++ simple_add.cu \
  --cuda-device-only \
  --cuda-gpu-arch=sm_61 \
  -S -o vectorAdd.ptx

# 查看 PTX 确认内核名
grep ".entry" vectorAdd.ptx
# 应该看到类似: .entry _Z9vectorAddPKfS0_Pfi(
```

#### 5.2 编译测试程序
```bash
# 编译 libptxrt
cd ../
mkdir build && cd build
cmake ..
make

# 编译测试程序（host-only）
clang++ ../examples/simple_add.cu \
  --cuda-host-only \
  -I.. \
  -L. \
  -lptxrt \
  -o simple_add
```

#### 5.3 运行测试
```bash
# 设置 PTX 路径
export PTXRT_PTX_PATH=../examples/vectorAdd.ptx

# 运行
./simple_add
```

#### 5.4 预期输出
```
[libptxrt] Fat binary registered
[libptxrt] Kernel registered: _Z9vectorAddPKfS0_Pfi at 0x...
[libptxrt] Launching kernel: _Z9vectorAddPKfS0_Pfi
[libptxrt] Loading PTX: ../examples/vectorAdd.ptx
[libptxrt] Kernel launched successfully
Vector addition successful! Verified 1024 elements.
```

---

## 🐛 常见问题排查

### 问题 1: 链接错误
```
undefined reference to `HostAPI::cuMemAlloc`
```
**解决**: 检查 CMakeLists.txt 是否正确链接了 PTX VM 库

### 问题 2: PTX 文件找不到
```
Failed to load PTX
```
**解决**: 
- 检查 PTX 文件路径
- 使用 `export PTXRT_PTX_PATH=/full/path/to/file.ptx`

### 问题 3: 内核名不匹配
```
Kernel not found
```
**解决**: 
- 查看 PTX 文件中的 `.entry` 名称
- 确保 `__cudaRegisterFunction` 记录的名称匹配

### 问题 4: CUfunction 为 0
**解决**: 需要实现 `cuModuleGetFunction` 从内核名获取函数句柄

---

## 📊 完成标准

- [x] 编译通过无警告
- [x] simple_add 能执行
- [x] 内存分配/释放工作正常
- [x] 内核能被调用
- [x] 结果验证通过

---

## ⏭️ 下一步优化（可选）

1. 实现真正的 Fat Binary 解析（自动提取 PTX）
2. 支持多个内核和多个 PTX 文件
3. 添加详细错误信息和调试日志
4. 支持更多参数类型（结构体、数组等）
5. 性能优化和内存池

**注意**: `<<<>>>` 语法已经通过实现 `cudaConfigureCall/cudaSetupArgument/cudaLaunch` 自动支持，Clang 会自动转换。

---

## 📞 需要帮助？

查看以下文档：
- `IMPLEMENTATION_QUICKSTART.md` - 详细实现指南
- `DEVELOPER_GUIDE.md` - API 参考
- `TODO.md` - 完整任务列表

**预计总时间**: 7-9 小时（含 <<<>>> 语法支持）
**难度**: ⭐⭐⭐ (中等)

🚀 开始实现吧！
