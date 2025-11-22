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