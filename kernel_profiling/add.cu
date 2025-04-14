#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define N 1024*32 // 1百万个元素
#define THREADS_PER_BLOCK 256

// 核函数声明
__global__ void addCoalesced(float* a, float* b, float* c, int n) {
    // 合并内存访问版本
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // 连续的内存访问模式
    if (i < n) {
        c[i] = a[i] + b[i];  // 相邻线程访问相邻内存地址
    }
}
__global__ void addNonCoalesced(float* a, float* b, float* c, int n) {
    // 非合并内存访问版本
    // 使用跨步访问模式来模拟非合并访问
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int gridStride = gridDim.x * blockDim.x;
    
    for (int i = bid * blockDim.x + tid; i < n; i += gridStride) {
        // 人为制造内存访问冲突和非合并访问
        int index = (i % 16) * (n / 16) + (i / 16);  // 打乱内存访问模式
        if (index < n) {
            c[index] = a[index] + b[index];  // 不连续的内存访问
        }
    }
}

void initializeVector(float* vec, int n) {
    for (int i = 0; i < n; i++) {
        vec[i] = (float)rand() / RAND_MAX;  // 初始化随机数
    }
}

int main() {
    float *a, *b, *c;          // 主机端指针
    float *d_a, *d_b, *d_c;     // 设备端指针
    size_t size = N * sizeof(float);
    
    // 分配主机内存
    a = (float*)malloc(size);
    b = (float*)malloc(size);
    c = (float*)malloc(size);
    
    initializeVector(a, N);
    initializeVector(b, N);
    
    // 分配设备内存
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);
    
    // 拷贝数据到设备
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);
    
    // 启动合并访问核函数
    dim3 block(THREADS_PER_BLOCK);
    dim3 grid((N + block.x - 1) / block.x);
    
    for (int i = 0; i < 2; i++) {
        addCoalesced<<<grid, block>>>(d_a, d_b, d_c, N);
        cudaDeviceSynchronize();
    }
    
    
    // 启动非合并访问核函数
    for (int i = 0; i < 2; i++) {
        addNonCoalesced<<<grid, block>>>(d_a, d_b, d_c, N);
        cudaDeviceSynchronize();
    }
    
    // 释放资源
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(a);
    free(b);
    free(c);
    
    return 0;
}