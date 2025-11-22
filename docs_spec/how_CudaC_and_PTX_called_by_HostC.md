CUDA 的 **Host (CPU) 与 Device (GPU)** 之间的数据交互过程：

* 第一部分是 “普通 CUDA C” 的方式（即用 `__global__` kernel）
* 第二部分是 “PTX（Parallel Thread Execution）” 方式（即通过 CUDA Driver API 载入和执行编译好的 PTX 汇编）

我来一步步讲清楚，并分别给出 **完整可运行 demo**。

---

## 🧩 一、Host C 调用 CUDA C 的方式（Runtime API）

### 💡思路

1. **Host 端（C）** 分配并初始化数据（在 CPU 上）。
2. **Device 端（GPU）** 分配显存，用 `cudaMalloc`。
3. **Host → Device** 拷贝输入数据，用 `cudaMemcpy`。
4. 启动 GPU kernel（`__global__` 函数）。
5. **Device → Host** 拷贝结果回来。
6. Host 打印或使用结果。

---

### ✅ 示例：向量加法（`host.cu`）

```c
#include <stdio.h>
#include <cuda_runtime.h>

// GPU 端 kernel
__global__ void vecAdd(const float *A, const float *B, float *C, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) C[idx] = A[idx] + B[idx];
}

int main() {
    int n = 8;
    size_t size = n * sizeof(float);

    // 分配并初始化 Host 内存
    float h_A[8], h_B[8], h_C[8];
    for (int i = 0; i < n; ++i) {
        h_A[i] = i;
        h_B[i] = i * 10;
    }

    // 分配 Device 内存
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    // 拷贝 Host → Device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // 启动 kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    vecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, n);

    // 拷回结果 Device → Host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // 打印结果
    printf("Result:\n");
    for (int i = 0; i < n; ++i)
        printf("%f + %f = %f\n", h_A[i], h_B[i], h_C[i]);

    // 释放内存
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return 0;
}
```

编译运行：

```bash
nvcc host.cu -o vecAdd
./vecAdd
```

这就是最常见的 “Host C 调用 CUDA C” 的交互方式。

---

## ⚙️ 二、Host 调用 PTX（使用 CUDA Driver API）

现在我们换成 **PTX 汇编代码**，不再直接写 `__global__` 函数。
流程是类似的，只是我们通过 Driver API（`cuModuleLoad`, `cuModuleGetFunction`, `cuLaunchKernel` 等）手动加载并执行。

---

### 🧠 思路对比

| 步骤        | CUDA C (Runtime API) | PTX (Driver API)                |
| --------- | -------------------- | ------------------------------- |
| 内存分配      | `cudaMalloc`         | `cuMemAlloc`                    |
| 内存拷贝      | `cudaMemcpy`         | `cuMemcpyHtoD` / `cuMemcpyDtoH` |
| 加载代码      | 编译在 .cu 中            | `cuModuleLoad` 从 `.ptx` 文件      |
| 获取 kernel | 编译器直接识别函数名           | `cuModuleGetFunction`           |
| 启动 kernel | `<<<>>>` 语法          | `cuLaunchKernel`                |

---

### ✅ 示例文件 1：PTX 文件（`vecAdd.ptx`）

（可由 nvcc 生成：`nvcc -ptx vecAdd.cu`）

```ptx
.version 8.0
.target sm_80
.address_size 64

.visible .entry vecAdd(
    .param .u64 A,
    .param .u64 B,
    .param .u64 C,
    .param .u32 N)
{
    .reg .u32 %r<6>;
    .reg .f32 %f<4>;
    .reg .pred %p;
    .reg .u64 %rd<10>;

    ld.param.u64 %rd1, [A];
    ld.param.u64 %rd2, [B];
    ld.param.u64 %rd3, [C];
    ld.param.u32 %r1, [N];

    mov.u32 %r2, %tid.x;
    mov.u32 %r3, %ctaid.x;
    mov.u32 %r4, %ntid.x;
    mad.lo.s32 %r5, %r3, %r4, %r2;
    setp.ge.u32 %p, %r5, %r1;
    @%p bra DONE;

    mul.wide.u32 %rd4, %r5, 4;
    add.u64 %rd5, %rd1, %rd4;
    add.u64 %rd6, %rd2, %rd4;
    add.u64 %rd7, %rd3, %rd4;

    ld.global.f32 %f1, [%rd5];
    ld.global.f32 %f2, [%rd6];
    add.f32 %f3, %f1, %f2;
    st.global.f32 [%rd7], %f3;

DONE:
    ret;
}
```

---

### ✅ 示例文件 2：Host C 加载 PTX 并执行（`host_ptx.c`）

```c
#include <stdio.h>
#include <cuda.h>

#define N 8

int main() {
    CUdevice device;
    CUcontext context;
    CUmodule module;
    CUfunction kernel;
    CUdeviceptr d_A, d_B, d_C;

    float h_A[N], h_B[N], h_C[N];
    for (int i = 0; i < N; i++) {
        h_A[i] = i;
        h_B[i] = i * 10;
    }

    // 初始化 CUDA Driver API
    cuInit(0);
    cuDeviceGet(&device, 0);
    cuCtxCreate(&context, 0, device);

    // 加载 PTX 模块
    cuModuleLoad(&module, "vecAdd.ptx");
    cuModuleGetFunction(&kernel, module, "vecAdd");

    // 分配 GPU 内存
    cuMemAlloc(&d_A, N * sizeof(float));
    cuMemAlloc(&d_B, N * sizeof(float));
    cuMemAlloc(&d_C, N * sizeof(float));

    // 拷贝 Host → Device
    cuMemcpyHtoD(d_A, h_A, N * sizeof(float));
    cuMemcpyHtoD(d_B, h_B, N * sizeof(float));

    // 设置参数
    void *args[] = { &d_A, &d_B, &d_C, (void*)&(int){N} };

    // 启动 kernel
    cuLaunchKernel(kernel,
                   1, 1, 1,      // grid dim
                   N, 1, 1,      // block dim
                   0, 0, args, 0);

    // 拷回结果
    cuMemcpyDtoH(h_C, d_C, N * sizeof(float));

    // 打印
    for (int i = 0; i < N; i++)
        printf("%f + %f = %f\n", h_A[i], h_B[i], h_C[i]);

    // 清理
    cuMemFree(d_A);
    cuMemFree(d_B);
    cuMemFree(d_C);
    cuModuleUnload(module);
    cuCtxDestroy(context);

    return 0;
}
```

编译与运行：

```bash
# 先编译 PTX
nvcc -ptx vecAdd.cu -o vecAdd.ptx
# 编译 host C
nvcc -o host_ptx host_ptx.c -lcuda
./host_ptx
```

输出与前面相同。

---

## 🧭 总结

| 操作        | Runtime API (CUDA C) | Driver API (PTX)              |
| --------- | -------------------- | ----------------------------- |
| 内存分配      | `cudaMalloc`         | `cuMemAlloc`                  |
| 内存拷贝      | `cudaMemcpy`         | `cuMemcpyHtoD / cuMemcpyDtoH` |
| 执行 kernel | `<<<>>>`             | `cuLaunchKernel`              |
| 加载代码      | 自动编译链接               | `cuModuleLoad` 从 `.ptx`       |
| 参数传递      | 直接传变量                | 参数数组（`void* args[]`）          |

Runtime API 更方便用于普通开发。
Driver API + PTX 适合底层控制、动态加载、运行时编译（如 JIT）。

---

是否希望我再给出一个 **host 通过加载字符串 PTX（不写文件）运行 kernel** 的例子？
这在实际中（比如动态生成 PTX）非常常见。
