





#include <iostream>
#include <cuda_runtime.h>
#include <cmath>
#include <chrono>


#define NUM 512


__global__ void matrixMulCUDA(float* C,  float* A,  float* B, int M, int N, int K) {
    // 考虑block得计算过程
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    float value = 0.0f;
    // float * A_ptr_start = A + blockDim.y * blockIdx.y * K ;
    // float * B_ptr_start = B + blockDim.x * blockIdx.x;

    for (int k = 0; k < K; k++) {
        // value += A_ptr_start[threadIdx.y * K + k] * B_ptr_start[k * N + threadIdx.x];
        value += A[y * K + k] * B[k * N + x];
    }
    C[y * N + x] = value;
    
}

// CUDA 核函数：矩阵乘法（全局内存）
// __global__ void matrixMulCUDA(float* C, const float* A, const float* B, int M, int N, int K) {
//     // 考虑block得计算过程
//     int row = blockIdx.y * blockDim.y + threadIdx.y; // 计算当前线程的行
//     int col = blockIdx.x * blockDim.x + threadIdx.x; // 计算当前线程的列

//     if (row < M && col < N) {
//         float value = 0.0f;
//         for (int i = 0; i < K; ++i) {
//             value += A[row * K + i] * B[i * N + col]; // 点积计算
//         }
//         C[row * N + col] = value; // 写入结果矩阵
//     }
// }



// CPU 实现：矩阵乘法
void matrixMulCPU(float* C, const float* A, const float* B, int M, int N, int K) {
    for (int row = 0; row < M; ++row) {
        for (int col = 0; col < N; ++col) {
            float value = 0.0f;
            for (int i = 0; i < K; ++i) {
                value += A[row * K + i] * B[i * N + col]; // 点积计算
            }
            C[row * N + col] = value; // 写入结果矩阵
        }
    }
}

// 比较两个矩阵的精度
bool compareMatrices(const float* C1, const float* C2, int M, int N, float epsilon = 1e-2) {
    for (int i = 0; i < M * N; ++i) {
        if (fabs(C1[i] - C2[i]) > epsilon) {
            std::cout << "Mismatch at index " << i << ": CUDA=" << C1[i] << ", CPU=" << C2[i] << std::endl;
            return false;
        }
    }
    return true;
}

int main() {
    int M = NUM * 2; // A 的行数
    int K = NUM; // A 的列数，B 的行数
    int N = NUM; // B 的列数

    // 分配主机内存
    float* h_A = new float[M * K];
    float* h_B = new float[K * N];
    float* h_C_cpu = new float[M * N];
    float* h_C_cuda = new float[M * N];

    // 初始化矩阵 A 和 B
    for (int i = 0; i < M * K; ++i) h_A[i] = static_cast<float>(rand()) / RAND_MAX;
    for (int i = 0; i < K * N; ++i) h_B[i] = static_cast<float>(rand()) / RAND_MAX;

    // 分配设备内存
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, K * N * sizeof(float));
    cudaMalloc(&d_C, M * N * sizeof(float));

    // 将数据从主机复制到设备
    cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice);

    // 定义 CUDA 线程块和网格大小
    dim3 blockSize(16, 16); // 每个线程块 16x16 线程
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (M + blockSize.y - 1) / blockSize.y);

    // 执行 CUDA 核函数
    auto start_cuda = std::chrono::high_resolution_clock::now();
    matrixMulCUDA<<<gridSize, blockSize>>>(d_C, d_A, d_B, M, N, K);
    cudaDeviceSynchronize(); // 等待 GPU 完成
    auto end_cuda = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_cuda = end_cuda - start_cuda;

    // 将结果从设备复制回主机
    cudaMemcpy(h_C_cuda, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    // 执行 CPU 实现
    auto start_cpu = std::chrono::high_resolution_clock::now();
    matrixMulCPU(h_C_cpu, h_A, h_B, M, N, K);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_cpu = end_cpu - start_cpu;

    // 比较 GPU 和 CPU 的结果
    if (compareMatrices(h_C_cuda, h_C_cpu, M, N)) {
        std::cout << "CUDA and CPU results match!" << std::endl;
    } else {
        std::cout << "CUDA and CPU results do not match!" << std::endl;
    }

    // 输出运行时间
    std::cout << "CUDA time: " << elapsed_cuda.count() << " seconds" << std::endl;
    std::cout << "CPU time: " << elapsed_cpu.count() << " seconds" << std::endl;

    // 释放内存
    delete[] h_A;
    delete[] h_B;
    delete[] h_C_cpu;
    delete[] h_C_cuda;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}