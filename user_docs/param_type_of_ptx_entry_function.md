> “PTX 的函数（尤其是 `.entry` 内核函数）到底能接收哪些类型的参数，以及这些参数在 Host ↔ Device 之间是如何传递的？”

我们来系统讲清楚这一点。

---

## 🧠 一、PTX 中的函数类型

PTX 有两类函数：

1. `.entry` —— **kernel 函数**（可从 Host 调用，用于 GPU 执行）
2. `.func` —— **普通 device 函数**（只能被其他 GPU 函数调用）

**参数规则主要针对 `.entry`。**

---

## 🚀 二、`.entry` 函数参数在 PTX 中的定义方式

PTX 中的 kernel 定义示例：

```ptx
.visible .entry myKernel(
    .param .u64 A,     // 指针参数
    .param .u32 N,     // 整数参数
    .param .f32 alpha  // 浮点参数
)
{
    // 内部代码...
}
```

这里的每个 `.param` 表示一个 **形参槽（parameter slot）**，
参数类型是基于 PTX 的数据类型系统。

---

## 🧩 三、PTX 可接收的参数类型类别

PTX 的 `.param` 可以是以下几类（本质上都是在 Host 启动 kernel 时传入的内存字节块）：

| 类型类别         | PTX 类型                                        | 典型用法             | Host 侧对应类型                                |
| ------------ | --------------------------------------------- | ---------------- | ----------------------------------------- |
| **标量整数**     | `.u8`, `.u16`, `.u32`, `.u64`, `.s32`, `.s64` | 用于传递索引、长度、标志等    | `unsigned int`, `int`, `long`, `size_t`   |
| **标量浮点**     | `.f16`, `.f32`, `.f64`                        | 用于浮点参数           | `float`, `double`                         |
| **指针类型**     | `.u64` (一般为64位地址)                             | 传递 GPU 全局内存地址    | `CUdeviceptr` 或 Runtime API 中的 `float*` 等 |
| **结构体/复合类型** | `.param .align` + `{}`                        | 可传复杂数据结构（需要对齐处理） | C struct（按字节序传入）                          |

---

## 🧮 四、Host 侧如何对应这些参数

以 CUDA Driver API (`cuLaunchKernel`) 为例：

```c
void *args[] = { &d_A, &N, &alpha };
```

每个指针对应 `.param` 的一个槽。

例如，对于下面的 PTX：

```ptx
.visible .entry scaleAdd(
    .param .u64 ptrA,
    .param .u32 N,
    .param .f32 alpha
)
```

Host 传参方式：

```c
CUdeviceptr d_A;
unsigned int N = 1024;
float alpha = 2.5f;

void *args[] = { &d_A, &N, &alpha };
cuLaunchKernel(kernel, 1,1,1, N,1,1, 0, 0, args, 0);
```

CUDA 驱动会自动将这些参数打包进 kernel 参数内存（param space），GPU 内部再通过 `ld.param.*` 指令取出。

---

## 🧱 五、在 PTX 内部如何访问这些参数

以 `.param .u64 A` 为例：

```ptx
.visible .entry myKernel(
    .param .u64 A,
    .param .u32 N
)
{
    .reg .u32 %r<3>;
    .reg .u64 %rd<3>;

    ld.param.u64 %rd1, [A];   // 取出指针参数 A
    ld.param.u32 %r1, [N];    // 取出标量参数 N
}
```

如果传入的是结构体，访问类似这样：

```ptx
.param .align 8 .b8 s[16];  // 结构体大小16字节
ld.param.v2.f32 {%f1, %f2}, [s+0]; // 读取结构体前8字节
```

---

## 📦 六、可以传结构体或数组吗？

可以，但有几点要注意：

### ✅ 结构体

PTX 支持 `.param .align N .b8 var[size];` 来定义结构体参数空间，
Host 侧可以传入结构体的字节拷贝（按字节序）。

例：

```ptx
.visible .entry kernelStruct(
    .param .align 8 .b8 config[16]
)
{
    .reg .f32 %f1, %f2;
    ld.param.f32 %f1, [config+0];
    ld.param.f32 %f2, [config+4];
}
```

Host 侧：

```c
struct Config { float a, b; };
Config cfg = {1.0f, 2.0f};
void *args[] = { &cfg };
```

### ⚠️ 数组

不能直接传“数组”类型，只能传“指针到数组”或“包含数组的结构体”。

> 因为参数是按值传递的字节块，不会自动展开。

---

## 🧰 七、特殊类型

| 类型                            | 描述             | 示例                               |
| ----------------------------- | -------------- | -------------------------------- |
| `.b8`, `.b16`, `.b32`, `.b64` | 原始字节（可用于任何类型）  | `.param .b32 flag`               |
| `.pred`                       | 布尔标志（很少用作参数）   | 不推荐直接传                           |
| `.texref`, `.surfref`         | 纹理、表面引用        | 由 CUDA 编译器内部生成                   |
| `.samplerref`                 | 采样器对象          | 同上                               |
| `.ptr`                        | 通用指针（新 PTX 支持） | `.param .ptr .global .align 8 A` |

---

## 🧩 八、总结对照表

| Host 类型                           | PTX 参数类型        | 说明            |
| --------------------------------- | --------------- | ------------- |
| `int` / `unsigned int`            | `.u32` / `.s32` | 32位整数         |
| `size_t` / `unsigned long long`   | `.u64`          | 64位整数或指针      |
| `float` / `double`                | `.f32` / `.f64` | 浮点参数          |
| `float*` / `int*` / `CUdeviceptr` | `.u64`          | GPU 全局内存指针    |
| `struct {...}`                    | `.b8 [N]` + 对齐  | 按字节拷贝结构体      |
| `bool`                            | `.pred`（不推荐）    | 通常用 `.u32` 替代 |

---

## 🔍 九、示例总结

PTX 侧：

```ptx
.visible .entry kernelDemo(
    .param .u64 A,
    .param .u32 N,
    .param .f32 alpha,
    .param .align 8 .b8 cfg[16]
)
```

Host 侧：

```c
CUdeviceptr d_A;
unsigned int N = 1000;
float alpha = 2.0f;
struct { float x, y, z, w; } cfg = {1, 2, 3, 4};

void *args[] = { &d_A, &N, &alpha, &cfg };
cuLaunchKernel(func, grid, 1,1, block,1,1, 0, 0, args, 0);
```

---
