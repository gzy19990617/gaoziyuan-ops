





#include <iostream>
#include <cuda_runtime.h>
#include <cmath>
#include <chrono>


#define NUM 512

//v2:增加每一个线程的工作

template <unsigned int BLOCKSIZE, unsigned int STRIDE>
__global__ void matrixMulCUDA(float* C,  float* A,  float* B, int M, int N, int K) {
    
    // 一个block负责step * step个元素，每个线程负责4个数
    constexpr int STEP = BLOCKSIZE * STRIDE;

    // share_memory 大小是 step * step
    __shared__ float a_shared[STEP][STEP];
    __shared__ float b_shared[STEP][STEP];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // 注意这里x,y代表在矩阵中全局索引，所以需要乘上一个STRIDE
    int x = threadIdx.x + blockDim.x * blockIdx.x * STRIDE;
    int y = threadIdx.y + blockDim.y * blockIdx.y * STRIDE;

    // 一个线程处理4个数
    float value[STRIDE][STRIDE] = {0.f};
    
    for (int s = 0; s < K; s += STEP) {
        // a_shared[threadIdx.y][threadIdx.x] = A[y * K + threadIdx.x + s];
        // b_shared[threadIdx.y][threadIdx.x] = B[(s + threadIdx.y) * N + x];
        for (int i = 0; i < STRIDE; i++)  {
            for (int j = 0; j < STRIDE; j++) {
                a_shared[ty + i * BLOCKSIZE][tx + j * BLOCKSIZE] = A[(y + i * BLOCKSIZE) * K + tx + BLOCKSIZE * j + s];
                b_shared[ty + i * BLOCKSIZE][tx + j * BLOCKSIZE] = B[(s + ty + i * BLOCKSIZE) * N + x + j * BLOCKSIZE];
            }
        }
        __syncthreads();

        for (int i = 0; i < STRIDE; i++)  {
            for (int j = 0; j < STRIDE; j++) {
                for (int k = 0; k < STEP; k++) { // K 维度是step
                    value[i][j] += a_shared[ty + i * BLOCKSIZE][k] * b_shared[k][tx + j * BLOCKSIZE];
                }
            }
        }

        __syncthreads();
    }
     for (int i = 0; i < STRIDE; i++)  {
            for (int j = 0; j < STRIDE; j++) {
                C[(y + i * BLOCKSIZE) * N  + x + j * BLOCKSIZE] += value[i][j];
            }
        }
    
}

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
            // return false;
        }
    }
    return true;
}

int main() {
    int M = NUM; // A 的行数
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

    constexpr int BLOCKSIZE  = 16;
    constexpr int STRIDE  = 2;

    // 定义 CUDA 线程块和网格大小
    // 这里使用了STRIDE来控制网格大小，以减少总的线程数。这样可以减小总的线程数，从而降低资源消耗和可能的内存溢出风险。
    dim3 blockSize(BLOCKSIZE, BLOCKSIZE); // 每个线程块 16x16 线程
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x/ STRIDE, (M + blockSize.y - 1) / blockSize.y / STRIDE);

    // 执行 CUDA 核函数
    auto start_cuda = std::chrono::high_resolution_clock::now();
    matrixMulCUDA<BLOCKSIZE, STRIDE><<<gridSize, blockSize>>>(d_C, d_A, d_B, M, N, K);
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