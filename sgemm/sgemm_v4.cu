
#include <iostream>
#include <cuda_runtime.h>
#include <cmath>
#include <chrono>


#define NUM 512

//v4:使用寄存器，两级缓存

// 提取k维度，从内积变为外积，其实就相当于对shread_memrory做分块
// 先拿一列，再拿一行，算出来一块
// 相当于从shared_memory中取的次数变少了

# define FETCH_FLOA4(pointer)(reinterpret_cast<float4 *>(&(pointer))[0])

template<unsigned int M_NUM_PER_BLOCK, unsigned int N_NUM_PER_BLOCK, unsigned int K_NUM_PER_BLOCK, unsigned int NUM_PER_THREAD>
__global__ void matrixMulCUDA(float* C,  float* A,  float* B, int M, int N, int K) {
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // reshape索引，便于寄存器取数据
    int tid = ty * blockDim.x + tx;
    int ctx = tid % 16;
    int cty = tid / 16;

    __shared__ float a_shared[M_NUM_PER_BLOCK][K_NUM_PER_BLOCK];
    __shared__ float b_shared[K_NUM_PER_BLOCK][N_NUM_PER_BLOCK];

    // 申请寄存器
    constexpr int REG_NUM = NUM_PER_THREAD / 2;
    float a_reg[REG_NUM] = {0.f};
    float b_reg[REG_NUM] = {0.f};

    float temp[REG_NUM][REG_NUM] = {0.f};

    int x = tx * NUM_PER_THREAD +  blockIdx.x * M_NUM_PER_BLOCK;
    int y = ty +  blockIdx.y * N_NUM_PER_BLOCK;
    
    // K方向做步进
    for (int s = 0; s < K; s += K_NUM_PER_BLOCK) {
        //可以根据最local版本方案去写索引
        // a_shared[threadIdx.y][threadIdx.x] = A[y * K + threadIdx.x + s];
        // b_shared[threadIdx.y][threadIdx.x] = B[(s + threadIdx.y) * N + x];
        
        FETCH_FLOA4(a_shared[ty][tx * NUM_PER_THREAD]) = FETCH_FLOA4(A[y * K + tx * NUM_PER_THREAD + s]);
        FETCH_FLOA4(b_shared[ty][tx * NUM_PER_THREAD]) = FETCH_FLOA4(B[N * (s + ty) + x]);
        __syncthreads();

        for (int k = 0; k < K_NUM_PER_BLOCK; k++) {
            a_reg[0] = a_shared[cty * 2][k];
            a_reg[1] = a_shared[cty * 2 + 1][k];
            b_reg[0] = b_shared[k][ctx * 2];
            b_reg[1] = b_shared[k][ctx * 2 + 1];
            for (int i = 0; i < REG_NUM; i++) {
                for (int j = 0; j < REG_NUM; j++) {
                    temp[i][j] += a_reg[i] * b_reg[j];
                }
            }
        }
        __syncthreads();
    }


    // 因为改变了线程排布，上面的全局索引用不了
    float * C_ptr_start = C + blockIdx.x * N_NUM_PER_BLOCK + blockIdx.y * M_NUM_PER_BLOCK * N;
    for (int i = 0; i < REG_NUM; i++) {
            for (int j = 0; j < REG_NUM; j++) {
                C_ptr_start[N *(cty * 2 + i) + ctx * 2 + j] = temp[i][j];
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
            return false;
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


    constexpr int M_NUM_PER_BLOCK = 32;
    constexpr int N_NUM_PER_BLOCK = 32;
    constexpr int K_NUM_PER_BLOCK = 32;
    constexpr int NUM_PER_THREAD = 4;

    // 定义 CUDA 线程块和网格大小
    // 这里使用了STRIDE来控制网格大小，以减少总的线程数。这样可以减小总的线程数，从而降低资源消耗和可能的内存溢出风险。
    dim3 blockSize(8, 32); // 每个线程块 16x16 线程
    dim3 gridSize(M / M_NUM_PER_BLOCK, N / N_NUM_PER_BLOCK);

    // 执行 CUDA 核函数
    auto start_cuda = std::chrono::high_resolution_clock::now();
    matrixMulCUDA<M_NUM_PER_BLOCK, N_NUM_PER_BLOCK, K_NUM_PER_BLOCK, NUM_PER_THREAD><<<gridSize, blockSize>>>(d_C, d_A, d_B, M, N, K);
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