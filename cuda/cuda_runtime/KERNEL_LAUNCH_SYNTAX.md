# CUDA `<<<>>>` 语法支持说明

## 问题
为什么 `<<<>>>` 语法需要实现？Clang 不是已经支持了吗？

## 答案
**Clang 只负责语法转换，实际执行需要我们实现对应的函数。**

---

## Clang 的工作

当你写：
```cpp
kernel<<<grid, block>>>(arg1, arg2, arg3);
```

Clang 编译器会自动转换为：
```cpp
if (cudaConfigureCall(grid, block, 0, 0) == cudaSuccess) {
    cudaSetupArgument(&arg1, sizeof(arg1), 0);
    cudaSetupArgument(&arg2, sizeof(arg2), sizeof(arg1));
    cudaSetupArgument(&arg3, sizeof(arg3), sizeof(arg1)+sizeof(arg2));
    cudaLaunch((void*)kernel);
}
```

---

## 我们的工作

**我们需要实现这三个函数**：

### 1. `cudaConfigureCall` - 保存配置
```cpp
cudaError_t cudaConfigureCall(dim3 gridDim, dim3 blockDim, 
                               size_t sharedMem, cudaStream_t stream);
```
作用：保存 grid/block 维度和其他配置到全局状态

### 2. `cudaSetupArgument` - 收集参数
```cpp
cudaError_t cudaSetupArgument(const void* arg, size_t size, size_t offset);
```
作用：将每个参数复制并保存到参数列表中

### 3. `cudaLaunch` - 实际启动
```cpp
cudaError_t cudaLaunch(const void* func);
```
作用：使用保存的配置和参数，调用 `cudaLaunchKernel()` 真正执行内核

---

## 调用流程

```
用户代码:
    kernel<<<1, 256>>>(a, b, c);
    
    ↓ (Clang 转换)
    
编译后:
    cudaConfigureCall(dim3(1,1,1), dim3(256,1,1), 0, NULL);
    cudaSetupArgument(&a, sizeof(a), 0);
    cudaSetupArgument(&b, sizeof(b), sizeof(a));
    cudaSetupArgument(&c, sizeof(c), sizeof(a)+sizeof(b));
    cudaLaunch((void*)kernel);
    
    ↓ (我们的实现)
    
实际执行:
    [cudaConfigureCall 保存 grid=1, block=256]
    [cudaSetupArgument 保存参数 a]
    [cudaSetupArgument 保存参数 b]
    [cudaSetupArgument 保存参数 c]
    [cudaLaunch 调用 cudaLaunchKernel(kernel, grid, block, {a,b,c})]
    
    ↓
    
    PTX VM 执行内核
```

---

## 实现状态

| 函数 | 声明 | 实现 | 状态 |
|-----|------|------|------|
| `cudaConfigureCall` | ✅ | ⏸️ | 需要实现 |
| `cudaSetupArgument` | ✅ | ⏸️ | 需要实现 |
| `cudaLaunch` | ✅ | ⏸️ | 需要实现 |
| `cudaLaunchKernel` | ✅ | ⏸️ | 需要实现 |

**结论**: 
- ✅ Clang 已经支持 `<<<>>>` **语法解析和转换**
- ⏸️ 我们需要实现对应的 **Runtime API 函数**
- ✅ 一旦实现了这三个函数，`<<<>>>` 就自动可用

---

## 测试示例

实现后，以下两种写法都能工作：

### 方式 1: 使用 `<<<>>>` (推荐)
```cpp
vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);
```

### 方式 2: 直接调用 API
```cpp
void* args[] = {&d_a, &d_b, &d_c, &N};
cudaLaunchKernel((void*)vectorAdd, 
                 dim3(blocksPerGrid), 
                 dim3(threadsPerBlock),
                 args, 0, nullptr);
```

两者完全等价，方式 1 更简洁，方式 2 更底层。

---

## 参考

- [CUDA Runtime API - cudaConfigureCall](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EXECUTION.html#group__CUDART__EXECUTION_1g6945d333289e2f4c667a7fb917c43e8c)
- [CUDA Runtime API - cudaSetupArgument](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EXECUTION.html#group__CUDART__EXECUTION_1g983f4bfb8b1e6c5c3b5c4b5b0d5c0c3e)
- [CUDA Runtime API - cudaLaunch](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EXECUTION.html#group__CUDART__EXECUTION_1gca6c2c5c1c9b4b4c4b4c4b4c4b4c4b4c)
- [Clang CUDA Support](https://llvm.org/docs/CompileCudaWithLLVM.html)
