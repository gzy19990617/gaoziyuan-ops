
#include <iostream>
#include <cuda_runtime.h>
#include <cmath>
#include <chrono>


#define NUM 512

//v7: double buffer

// global memory搬运到shared memory 大循环，每一次里面的运算，小循环

# define FETCH_FLOA4(pointer)(reinterpret_cast<float4 *>(&(pointer))[0])

template <const int BLOCK_SIZE_M, // 每一个block计算的C中的block的height
          const int BLOCK_SIZE_N,
          const int BLOCK_SIZE_K,
          const int THREAD_SIZE_Y,
          const int THREAD_SIZE_X,
          const bool ENABLE_DOUBLE_BUFFER>
__global__ void matrixMulCUDA(float* C,  float* A,  float* B, int M, int N, int K) {
    
    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // thread index in a block
    const int tid = ty * blockDim.x + tx;

    // 申请两组寄存器
    __shared__ float a_shared[2][BLOCK_SIZE_K][BLOCK_SIZE_M];
    __shared__ float b_shared[2][BLOCK_SIZE_K][BLOCK_SIZE_N];

    // 一个线程处理64个数
    float accum[THREAD_SIZE_Y][THREAD_SIZE_X] = {0.f};
    float reg_a[THREAD_SIZE_Y] = {0.f};
    float reg_b[THREAD_SIZE_X] = {0.f};

    // 用于转置
    float ldg_a_reg[4] = {0.f};

    float *A_ptr_start = A + blockIdx.y * BLOCK_SIZE_M * K;
    float *B_ptr_start = B + blockIdx.x * BLOCK_SIZE_N;

    // A\B tile 每行需要几个线程
    const int A_tile_thread_per_row = BLOCK_SIZE_K / 4; // 2
    const int B_tile_thread_per_row = BLOCK_SIZE_N / 4; // 32

    // 线程重排后的索引
    const int A_tile_tid_x = tid % A_tile_thread_per_row;
    const int A_tile_tid_y = tid / A_tile_thread_per_row;

    const int B_tile_tid_x = tid % B_tile_thread_per_row;
    const int B_tile_tid_y = tid / B_tile_thread_per_row;

    // A share_memory
    FETCH_FLOA4(ldg_a_reg[0]) = FETCH_FLOA4(A_ptr_start[A_tile_tid_y * K + A_tile_tid_x * 4]); // 一个线程处理四个数
    // 转置
    a_shared[0][A_tile_tid_x * 4][A_tile_tid_y] = ldg_a_reg[0];
    a_shared[0][A_tile_tid_x * 4 + 1][A_tile_tid_y] = ldg_a_reg[1];
    a_shared[0][A_tile_tid_x * 4 + 2][A_tile_tid_y] = ldg_a_reg[2];
    a_shared[0][A_tile_tid_x * 4 + 3][A_tile_tid_y] = ldg_a_reg[3];

    // B share_memory
    FETCH_FLOA4(b_shared[0][B_tile_tid_y][B_tile_tid_x * 4]) = FETCH_FLOA4(B_ptr_start[B_tile_tid_y * N + B_tile_tid_x * 4]); // 一个线程处理四个数

    __syncthreads();

    int write_stage_idx = 1;
    for (int s = BLOCK_SIZE_K; s < K; s += BLOCK_SIZE_K) { // 从第一个开始取

        // 流水线的第一阶段
        // A share_memory
        FETCH_FLOA4(ldg_a_reg[0]) = FETCH_FLOA4(A_ptr_start[A_tile_tid_y * K + A_tile_tid_x * 4 + s]); // 一个线程处理四个数
        // 转置
        a_shared[write_stage_idx][A_tile_tid_x * 4][A_tile_tid_y] = ldg_a_reg[0];
        a_shared[write_stage_idx][A_tile_tid_x * 4 + 1][A_tile_tid_y] = ldg_a_reg[1];
        a_shared[write_stage_idx][A_tile_tid_x * 4 + 2][A_tile_tid_y] = ldg_a_reg[2];
        a_shared[write_stage_idx][A_tile_tid_x * 4 + 3][A_tile_tid_y] = ldg_a_reg[3];

        // B share_memory
        FETCH_FLOA4(b_shared[write_stage_idx][B_tile_tid_y][B_tile_tid_x * 4]) = FETCH_FLOA4(B_ptr_start[(B_tile_tid_y + s) * N + B_tile_tid_x * 4]); // 一个线程处理四个数

        // 流水线的第二阶段运算
        write_stage_idx = write_stage_idx ^ 1; // 取反
        // 提取K维度
        for (int k = 0; k < BLOCK_SIZE_K; k++) {
            // ty 变为原始索引
            FETCH_FLOA4(reg_a[0]) = FETCH_FLOA4(a_shared[write_stage_idx][k][ty * THREAD_SIZE_Y]);
            FETCH_FLOA4(reg_a[4]) = FETCH_FLOA4(a_shared[write_stage_idx][k][ty * THREAD_SIZE_Y + 4]);
            FETCH_FLOA4(reg_b[0]) = FETCH_FLOA4(b_shared[write_stage_idx][k][tx * THREAD_SIZE_X]);
            FETCH_FLOA4(reg_b[4]) = FETCH_FLOA4(b_shared[write_stage_idx][k][tx * THREAD_SIZE_X + 4]);

            for (int i = 0; i < THREAD_SIZE_Y; i++) {
                for (int j = 0; j < THREAD_SIZE_X; j++) {
                    accum[i][j] += reg_a[i] * reg_b[j];
                }
            }
        }
         __syncthreads();
    }

    // 最后一个多余的
    write_stage_idx = write_stage_idx ^ 1; // 取反
    for (int k = 0; k < BLOCK_SIZE_K; k++) {
        // ty 变为原始索引
        FETCH_FLOA4(reg_a[0]) = FETCH_FLOA4(a_shared[write_stage_idx][k][ty * THREAD_SIZE_Y]);
        FETCH_FLOA4(reg_a[4]) = FETCH_FLOA4(a_shared[write_stage_idx][k][ty * THREAD_SIZE_Y + 4]);
        FETCH_FLOA4(reg_b[0]) = FETCH_FLOA4(b_shared[write_stage_idx][k][tx * THREAD_SIZE_X]);
        FETCH_FLOA4(reg_b[4]) = FETCH_FLOA4(b_shared[write_stage_idx][k][tx * THREAD_SIZE_X + 4]);

        for (int i = 0; i < THREAD_SIZE_Y; i++) {
            for (int j = 0; j < THREAD_SIZE_X; j++) {
                accum[i][j] += reg_a[i] * reg_b[j];
            }
        }
    }

    // 将结果写回全局内存
    float* C_ptr_start = C + N * by * BLOCK_SIZE_M + bx * BLOCK_SIZE_N;
    for(int i = 0; i < THREAD_SIZE_Y; i++) {
        FETCH_FLOA4(C_ptr_start[N * (ty * THREAD_SIZE_Y + i) + tx * THREAD_SIZE_X]) = FETCH_FLOA4(accum[i][0]);
        FETCH_FLOA4(C_ptr_start[N * (ty * THREAD_SIZE_Y + i) + tx * THREAD_SIZE_X + 4]) = FETCH_FLOA4(accum[i][4]);
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


    constexpr int BLOCK_SIZE_M = 128;
    constexpr int BLOCK_SIZE_K = 8;
    constexpr int BLOCK_SIZE_N = 128;

    constexpr int THREAD_SIZE_X = 8;
    constexpr int THREAD_SIZE_Y = 8;

    constexpr bool ENABLE_DOUBLE_BUFFER = true;


    // dim3 blockSize(16, 16); 
    dim3 blockSize(BLOCK_SIZE_N / THREAD_SIZE_X, BLOCK_SIZE_M / THREAD_SIZE_Y);
    dim3 gridSize( N / BLOCK_SIZE_N, M / BLOCK_SIZE_M);

    // 执行 CUDA 核函数
    auto start_cuda = std::chrono::high_resolution_clock::now();
    matrixMulCUDA<BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, THREAD_SIZE_Y, THREAD_SIZE_X, ENABLE_DOUBLE_BUFFER><<<gridSize, blockSize>>>(d_C, d_A, d_B, M, N, K);
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