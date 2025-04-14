#include <iostream>
#include <vector>
#include <cstdlib>
#include <cassert>
#include <chrono>

// CPU转置：逐行读取，逐列写入
void transpose_cpu(float* out, const float* in, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            out[j * rows + i] = in[i * cols + j];  // 行优先→列优先
        }
    }
}

__global__ void transpose_gpu(float* out, const float* in, int rows, int cols) {
    // 计算全局索引（合并访问out，非合并访问in）
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < cols && y < rows) {
        out[x * rows + y] = in[y * cols + x];  // 转置操作
    }
}

bool check_result(const float* cpu_out, const float* gpu_out, int size) {
    const float eps = 1e-5;
    for (int i = 0; i < size; ++i) {
        if (fabs(cpu_out[i] - gpu_out[i]) > eps) {
            printf("Mismatch at %d: CPU=%.2f, GPU=%.2f\n", 
                   i, cpu_out[i], gpu_out[i]);
            return false;
        }
    }
    return true;
}

void launch_transpose_gpu(int block_x, int block_y, float* d_out, float* d_in, int rows, int cols) {
    dim3 block(block_x, block_y);  // 每个block 256线程
    dim3 grid((cols + block.x - 1) / block.x, 
             (rows + block.y - 1) / block.y);
    
    // 创建CUDA事件用于计时
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // 记录开始时间
    cudaEventRecord(start);
    
    transpose_gpu<<<grid, block>>>(d_out, d_in, rows, cols);
    
    // 记录结束时间
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    // 计算耗时
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    std::cout << "GPU Kernel execution time: " << milliseconds * 1000 << " us" << std::endl;
    
    // 清理事件
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    cudaDeviceSynchronize();
}

int main() {
    const int rows = 2048, cols = 512;
    const int size = rows * cols;

    // 分配主机内存
    std::vector<float> h_in(size), h_cpu_out(size), h_gpu_out(size);
    for (int i = 0; i < size; ++i) {
        h_in[i] = static_cast<float>(rand()) / RAND_MAX;  // 随机初始化
    }

    // CPU转置并计时
    auto cpu_start = std::chrono::high_resolution_clock::now();
    transpose_cpu(h_cpu_out.data(), h_in.data(), rows, cols);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_time = cpu_end - cpu_start;
    std::cout << "CPU execution time: " << cpu_time.count() << " ms" << std::endl;

    // GPU转置
    float *d_in, *d_out;
    cudaMalloc(&d_in, size * sizeof(float));
    cudaMalloc(&d_out, size * sizeof(float));
    cudaMemcpy(d_in, h_in.data(), size * sizeof(float), cudaMemcpyHostToDevice);
    for (int i = 0; i < 5; i++) {
        launch_transpose_gpu(32, 8, d_out, d_in, rows, cols);
    }

    // 合并程度 1/8
    for (int i = 0; i < 5; i++) {
        launch_transpose_gpu(16, 16, d_out, d_in, rows, cols);
    }

    // 合并程度 4/8
    // 至少8个线程可以合并一个section
    for (int i = 0; i < 5; i++) {
        launch_transpose_gpu(8, 32, d_out, d_in, rows, cols);
    }

    for (int i = 0; i < 5; i++) {
        launch_transpose_gpu(4, 64, d_out, d_in, rows, cols);
    }
    cudaMemcpy(h_gpu_out.data(), d_out, size * sizeof(float), cudaMemcpyDeviceToHost);

    // 验证结果
    if (check_result(h_cpu_out.data(), h_gpu_out.data(), size)) {
        std::cout << "Result check PASSED!" << std::endl;
    } else {
        std::cerr << "Result check FAILED!" << std::endl;
    }

    // 释放资源
    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}