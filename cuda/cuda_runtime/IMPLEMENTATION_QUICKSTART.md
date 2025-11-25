# libptxrt 实现指南 - 快速上手

## 🎯 目标
让 `examples/simple_add.cu` 能够通过 libptxrt 在 PTX VM 上成功运行。

---

## 📚 现有 PTX VM API 分析

根据 `include/host_api.hpp` 和 `include/vm.hpp`，PTX VM 提供了以下接口：

### HostAPI 类（Driver API 风格）
```cpp
class HostAPI {
    // 程序加载
    bool loadProgram(const std::string& filename);
    
    // 内存管理
    CUresult cuMemAlloc(CUdeviceptr* dptr, size_t bytesize);
    CUresult cuMemFree(CUdeviceptr dptr);
    CUresult cuMemcpyHtoD(CUdeviceptr dstDevice, const void* srcHost, size_t ByteCount);
    CUresult cuMemcpyDtoH(void* dstHost, CUdeviceptr srcDevice, size_t ByteCount);
    
    // 内核执行
    CUresult cuLaunchKernel(CUfunction f,
                          unsigned int gridDimX, gridDimY, gridDimZ,
                          unsigned int blockDimX, blockDimY, blockDimZ,
                          unsigned int sharedMemBytes, CUstream hStream,
                          void** kernelParams, void** extra);
};
```

### PTXVM 类（底层接口）
```cpp
class PTXVM {
    // 程序加载
    bool loadProgram(const std::string& filename);
    
    // 内核设置
    void setKernelName(const std::string& name);
    void setKernelLaunchParams(const KernelLaunchParams& params);
    void setKernelParameters(const std::vector<KernelParameter>& parameters);
    
    // 内存管理
    CUdeviceptr allocateMemory(size_t size);
    bool freeMemory(CUdeviceptr ptr);
    bool copyMemoryHtoD(CUdeviceptr dst, const void* src, size_t size);
    bool copyMemoryDtoH(void* dst, CUdeviceptr src, size_t size);
    
    // 执行
    bool run();
};
```

---

## 🚀 实现策略

### 方案 A：使用 HostAPI（推荐）
优点：接口完整，类似 CUDA Driver API  
缺点：需要先将 PTX 保存为文件

### 方案 B：使用 PTXVM 直接
优点：可以直接加载 PTX 字符串（如果支持）  
缺点：接口较底层

**推荐使用方案 A**，因为 HostAPI 更成熟。

---

## 📝 详细实现步骤

### 步骤 1：增强 RuntimeState（内部状态管理）

在 `cuda_runtime_internal.h` 中已有框架，需要添加：

```cpp
class RuntimeState {
public:
    // 添加 HostAPI 实例
    std::unique_ptr<HostAPI> host_api;
    
    // PTX 代码临时文件路径映射
    std::map<std::string, std::string> kernel_ptx_files;
    
    RuntimeState() {
        host_api = std::make_unique<HostAPI>();
        host_api->initialize();
        host_api->cuInit(0);
    }
};
```

### 步骤 2：实现 Fat Binary 解析

**方式 1：手动解析（复杂）**
```cpp
void** __cudaRegisterFatBinary(void* fatCubin) {
    // 1. 解析 wrapper
    struct __fatBinC_Wrapper_t {
        int magic;
        int version;
        void* data;
        void* filename_or_fatbins;
    };
    
    auto wrapper = reinterpret_cast<__fatBinC_Wrapper_t*>(fatCubin);
    
    // 2. 查找 PTX section
    // 需要逆向 fat binary 格式...
}
```

**方式 2：使用 Clang 直接生成 PTX（推荐）**
```bash
# 编译时直接生成 PTX 文件，不依赖 fat binary
clang++ simple_add.cu \
  --cuda-device-only \
  --cuda-gpu-arch=sm_61 \
  -S -o simple_add.ptx

# 然后在代码中直接加载 PTX 文件
```

**简化实现（初期）**：
```cpp
void** __cudaRegisterFatBinary(void* fatCubin) {
    // 暂时返回一个假的 handle
    // 用户需要手动提供 PTX 文件
    static void* dummy_handle = malloc(8);
    printf("[libptxrt] Fat binary registered, please provide PTX file manually\n");
    return &dummy_handle;
}

void __cudaRegisterFunction(void** fatCubinHandle,
                           const char* hostFun,
                           char* deviceFun,
                           const char* deviceName, ...) {
    // 记录内核名称
    auto& state = RuntimeState::getInstance();
    KernelInfo info;
    info.kernel_name = deviceName;
    info.host_func = (const void*)hostFun;
    
    // 存储映射
    state.kernel_map[(const void*)hostFun] = info;
    
    printf("[libptxrt] Kernel registered: %s at %p\n", deviceName, hostFun);
}
```

### 步骤 3：实现内存管理

```cpp
cudaError_t cudaMalloc(void** devPtr, size_t size) {
    auto& state = RuntimeState::getInstance();
    
    // 使用 HostAPI 分配内存
    CUdeviceptr ptr;
    CUresult result = state.host_api->cuMemAlloc(&ptr, size);
    
    if (result != CUDA_SUCCESS) {
        return cudaErrorMemoryAllocation;
    }
    
    // 保存映射
    *devPtr = reinterpret_cast<void*>(ptr);
    state.device_memory[*devPtr] = {*devPtr, size, false};
    
    return cudaSuccess;
}

cudaError_t cudaFree(void* devPtr) {
    auto& state = RuntimeState::getInstance();
    
    CUdeviceptr ptr = reinterpret_cast<CUdeviceptr>(devPtr);
    CUresult result = state.host_api->cuMemFree(ptr);
    
    state.device_memory.erase(devPtr);
    
    return (result == CUDA_SUCCESS) ? cudaSuccess : cudaErrorUnknown;
}

cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, 
                       enum cudaMemcpyKind kind) {
    auto& state = RuntimeState::getInstance();
    CUresult result;
    
    switch (kind) {
        case cudaMemcpyHostToDevice:
            result = state.host_api->cuMemcpyHtoD(
                reinterpret_cast<CUdeviceptr>(dst), src, count);
            break;
            
        case cudaMemcpyDeviceToHost:
            result = state.host_api->cuMemcpyDtoH(
                dst, reinterpret_cast<CUdeviceptr>(src), count);
            break;
            
        case cudaMemcpyDeviceToDevice:
            // 需要实现 DtoD
            return cudaErrorNotSupported;
            
        default:
            return cudaErrorInvalidMemcpyDirection;
    }
    
    return (result == CUDA_SUCCESS) ? cudaSuccess : cudaErrorUnknown;
}
```

### 步骤 4：实现内核启动（关键）

```cpp
cudaError_t cudaLaunchKernel(const void* func, dim3 gridDim, dim3 blockDim,
                             void** args, size_t sharedMem, cudaStream_t stream) {
    auto& state = RuntimeState::getInstance();
    
    // 1. 查找内核信息
    auto it = state.kernel_map.find(func);
    if (it == state.kernel_map.end()) {
        fprintf(stderr, "[libptxrt] Kernel not found for function %p\n", func);
        return cudaErrorInvalidValue;
    }
    
    const KernelInfo& kernel = it->second;
    printf("[libptxrt] Launching kernel: %s\n", kernel.kernel_name.c_str());
    
    // 2. 确保 PTX 程序已加载
    // 方式 A: 从文件加载（需要先保存 PTX）
    std::string ptx_file = kernel.kernel_name + ".ptx";
    if (!state.host_api->isProgramLoaded()) {
        if (!state.host_api->loadProgram(ptx_file)) {
            fprintf(stderr, "[libptxrt] Failed to load PTX: %s\n", ptx_file.c_str());
            return cudaErrorInvalidSource;
        }
    }
    
    // 3. 调用 cuLaunchKernel
    // 注意：需要获取 CUfunction handle（这里需要扩展 HostAPI）
    CUfunction f = 0; // 需要从 kernel_name 查找
    
    CUresult result = state.host_api->cuLaunchKernel(
        f,
        gridDim.x, gridDim.y, gridDim.z,
        blockDim.x, blockDim.y, blockDim.z,
        sharedMem,
        nullptr,  // stream
        args,     // kernel parameters
        nullptr   // extra
    );
    
    return (result == CUDA_SUCCESS) ? cudaSuccess : cudaErrorLaunchFailure;
}
```

---

## 🔧 快速原型实现（最小工作版本）

### 1. 修改 HostAPI 增加内核查找
需要在 `HostAPI` 中添加：
```cpp
// 在 host_api.hpp 中添加
CUresult cuModuleGetFunction(CUfunction* hfunc, const char* name);
```

### 2. 创建辅助脚本自动提取 PTX
```bash
#!/bin/bash
# extract_ptx.sh

CUDA_FILE=$1
PTX_FILE="${CUDA_FILE%.cu}.ptx"

clang++ "$CUDA_FILE" \
  --cuda-device-only \
  --cuda-gpu-arch=sm_61 \
  -S -o "$PTX_FILE"

echo "PTX extracted to $PTX_FILE"
```

### 3. 使用流程
```bash
# 1. 提取 PTX
./extract_ptx.sh simple_add.cu  # 生成 simple_add.ptx

# 2. 编译主机代码链接 libptxrt
clang++ simple_add.cu \
  --cuda-host-only \
  -I../cuda/cuda_runtime \
  -L../cuda/cuda_runtime/build \
  -lptxrt \
  -o simple_add

# 3. 运行（PTX 文件需要在同目录）
./simple_add
```

---

## ⚡ 最小实现检查清单

- [ ] **步骤 1**: 链接 HostAPI 到 libptxrt
  - [ ] 在 `cuda_runtime/CMakeLists.txt` 中添加依赖
  - [ ] 包含 `host_api.hpp`
  
- [ ] **步骤 2**: 实现内存管理（3个函数）
  - [ ] `cudaMalloc()` → `cuMemAlloc()`
  - [ ] `cudaFree()` → `cuMemFree()`
  - [ ] `cudaMemcpy()` → `cuMemcpyHtoD/DtoH()`
  
- [ ] **步骤 3**: 实现内核注册（2个函数）
  - [ ] `__cudaRegisterFatBinary()` - 简化版本
  - [ ] `__cudaRegisterFunction()` - 建立映射表
  
- [ ] **步骤 4**: 实现内核启动
  - [ ] `cudaLaunchKernel()` → `cuLaunchKernel()`
  - [ ] 处理参数传递
  
- [ ] **步骤 5**: 测试
  - [ ] 手动提取 PTX
  - [ ] 编译链接测试程序
  - [ ] 运行并验证结果

---

## 🎓 实现建议

### 初期简化方案
1. **跳过 Fat Binary 解析**：要求用户手动提供 PTX 文件
2. **使用环境变量**：通过 `PTXRT_PTX_PATH` 指定 PTX 文件路径
3. **单内核假设**：假设程序只有一个内核

### 调试技巧
```cpp
// 在关键函数中添加日志
#define PTXRT_LOG(fmt, ...) \
    fprintf(stderr, "[libptxrt] " fmt "\n", ##__VA_ARGS__)

cudaError_t cudaMalloc(void** devPtr, size_t size) {
    PTXRT_LOG("cudaMalloc: %zu bytes", size);
    // ... 实现
}
```

### 参数传递示例
```cpp
// simple_add 的参数：vectorAdd(float *a, float *b, float *c, int n)
void** args = {&d_a, &d_b, &d_c, &n};

// 在 cuLaunchKernel 中，这些参数会被传递给 PTX VM
```

---

## ⏱️ 预计时间

| 任务 | 时间 | 难度 |
|-----|------|------|
| 链接 HostAPI | 30分钟 | ⭐ |
| 实现内存管理 | 1小时 | ⭐⭐ |
| 实现内核注册 | 1小时 | ⭐⭐ |
| 实现内核启动 | 2小时 | ⭐⭐⭐ |
| 调试测试 | 2-3小时 | ⭐⭐⭐ |
| **总计** | **6-8小时** | |

---

## 📌 下一步行动

1. **立即开始**：修改 `cuda_runtime/CMakeLists.txt` 添加 HostAPI 依赖
2. **先实现内存**：因为最简单，可以快速验证链接是否正确
3. **再实现启动**：这是核心功能
4. **最后完善注册**：可以先用硬编码测试

需要我开始实现吗？
