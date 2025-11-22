实现 CUDA→PTX 虚拟机运行的方式。
总体思路：

> 🔧 **实现一套与 CUDA Runtime API 同名、同参数的接口（libptxrt.so / .a）**，
> 然后在编译链接阶段，用它替代 NVIDIA 的 `libcudart`，
> 就能让用户的标准 CUDA 程序“假装”跑在 GPU 上，
> 实际上由 **PTX 虚拟机执行。**

---

## ✅ 一、整体机制

当 Clang / NVCC 编译 CUDA 程序时：

* 生成普通的 **x86 主机代码**；
* 在 `.nv_fatbin` 段中嵌入 **PTX 文本**；
* 并在 host 代码里**调用一系列 CUDA Runtime 函数**（如 `cudaMalloc`、`cudaLaunchKernel` 等）。

要做的：

1. 实现这些函数；
2. 把它们放进 `libptxrt`；
3. 用户在链接时用库 `libptxrt` 代替 NVIDIA 的库。

---

## ⚙️ 二、编译 & 链接流程示例

### 1️⃣ 用户源代码（完全标准 CUDA）
TODO：complete below demo code
```cpp
// cuda/cudac/cudac_demo.cu
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void add(float *a, float *b, float *c) {
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

int main() {
    const int N = 4;
    float h_a[N] = {1.0f, 2.0f, 3.0f, 4.0f};
    float h_b[N] = {5.0f, 6.0f, 7.0f, 8.0f};
    float h_c[N] = {0};

    float *d_a, *d_b, *d_c;
    
    // 分配设备内存
    cudaMalloc(&d_a, N * sizeof(float));
    cudaMalloc(&d_b, N * sizeof(float));
    cudaMalloc(&d_c, N * sizeof(float));
    
    // 拷贝数据到设备
    cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice);
    
    // 执行 kernel
    add<<<1, N>>>(d_a, d_b, d_c);
    
    // 同步
    cudaDeviceSynchronize();
    
    // 拷贝结果回主机
    cudaMemcpy(h_c, d_c, N * sizeof(float), cudaMemcpyDeviceToHost);
    
    // 打印结果
    printf("Results:\n");
    for (int i = 0; i < N; i++) {
        printf("%.1f + %.1f = %.1f\n", h_a[i], h_b[i], h_c[i]);
    }
    
    // 释放内存
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    
    return 0;
}
```

### 2️⃣ 用 clang 编译
TODO: which sm version is corresponding to ptx 6.1 which is the supported ptx version?
```bash
clang++ add.cu \
  --cuda-path=/usr/local/cuda \
  --cuda-gpu-arch=sm_80 \
  -O2 -L. -lptxrt -o add
```

> 注意：**不再链接 `-lcudart`**。
> `-lptxrt` 是替代库。

---

## 🧩 三、要实现的内容（libptxrt）

需要实现：

| 函数                            | 作用                      |
| ----------------------------- | ----------------------- |
| `__cudaRegisterFatBinary()`   | 解析 `.nv_fatbin` 段中的 PTX |
| `__cudaRegisterFunction()`    | 记录 kernel 名 ↔ 函数地址      |
| `cudaMalloc()` / `cudaFree()` | 模拟 GPU 内存（可用 malloc）    |
| `cudaMemcpy()`                | 模拟 Host↔Device 拷贝       |
| `cudaLaunchKernel()`          | 调用 ptx-vm 执行 PTX      |
| `cudaDeviceSynchronize()`     | 空操作或等待虚拟机执行完            |

---

## 🧠 四、执行机制概览

当程序启动时：

1. Clang 自动插入的初始化函数调用：

   ```cpp
   __cudaRegisterFatBinary();
   __cudaRegisterFunction();
   ```

   →  `libptxrt` 会接收到 `.nv_fatbin` 中的 PTX 文本，并存储。

2. 当执行 `add<<<1,4>>>(...)` 时：

   * Clang 会生成 host stub 调用 `cudaLaunchKernel()`；
   * 查表找到对应的 PTX；
   * 调用 `ptx_vm_run(ptx, kernel_name, grid, block, args)`；
   * PTX 虚拟机解释执行内核。

3. 所有内存和参数都在 x86 上模拟。

---

## 🔍 五、示意图

```
┌────────────────────────────┐
│ User CUDA Code (.cu)       │
│ ├── __global__ kernels     │
│ └── cudaMalloc, <<<>>>     │
└──────────┬─────────────────┘
           │ clang++ compile
           ▼
┌────────────────────────────┐
│ Executable (x86 ELF)       │
│ ├── .text (host code)      │
│ ├── .nv_fatbin (PTX)       │
│ └── linked to libptxrt     │
└──────────┬─────────────────┘
           │
           ▼
┌────────────────────────────┐
│ libptxrt                   │
│ ├── implements CUDA API    │
│ ├── parse fatbin → PTX     │
│ ├── simulate cudaMalloc    │
│ ├── cudaLaunchKernel → ptx-vm │
│ └── interface to ptx-vm    │
└────────────────────────────┘
```

---

## ⚙️ 六、关键点与注意事项

| 重点               | 说明                            |
| ---------------- | ----------------------------- |
| `.nv_fatbin` 段存在 | 由 clang 自动生成，可从 ELF 中读出       |
| PTX 提取           | 直接搜索 `".version"` 到 `"exit;"` |
| 内存模型             | 可直接用 malloc/free 模拟           |
| 多线程模型            | 可由虚拟机内部模拟 block/thread        |
| 同步语义             | `cudaDeviceSynchronize()` 可为空 |
| 兼容性              | 大多数简单 CUDA 程序可直接运行            |
| 限制               | 不支持 cubin（已编译 SASS），只支持 PTX   |

---

## ✅ 七、最终结论

| 项目                 | 是否需要             |
| ------------------ | ---------------- |
| 修改用户代码             | ❌ 不需要            |
| 修改编译器              | ❌ 不需要            |
| 替换 CUDA runtime 库  | ✅ 必须（用 libptxrt） |
| 自己解析 fatbin 中的 PTX | ✅ 必须             |
| 使用 ptx-vm 解释执行     | ✅ 必须             |

> ✅ **所以：**
> 实现一份完整的 CUDA Runtime API 替代库（`libptxrt`），
> 并让用户在编译时链接它而不是 `libcudart`，
> 就能让标准 CUDA 程序**在 x86 上直接执行**，
> 所有 GPU 调用都自动走 **PTX 虚拟机**。

---

* `libptxrt` 的最小可编译框架（含 `__cudaRegisterFatBinary`、`cudaLaunchKernel` 等 stub），
* 并展示如何调用 `ptx-vm` 接口执行 PTX，
