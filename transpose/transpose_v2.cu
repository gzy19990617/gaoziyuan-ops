#include <iostream>
#include <vector>
#include <cstdlib>
#include <cassert>
#include <chrono>

// 一个线程负责4*4个元素，在寄存器内进行转置

// 每个线程都有自己独立的寄存器组，寄存器中的数据对其他线程是不可见的。因为寄存器是线程私有的，所以不存在多个线程同时访问同一寄存器的情况，自然不需要同步。

# define FETCH_FLOA4(pointer)(reinterpret_cast<float4 *>(&(pointer))[0])

template<unsigned int Y_NUM_PER_THREAD, unsigned int X_NUM_PER_THREAD>
__global__ void transpose_gpu(float* out, float* in, int rows, int cols) {
    
    float src_transpose[4][4]; // 这些是寄存器变量
    float dst_transpose[4][4]; // 每个线程独立拥有

    int x = blockIdx.x * blockDim.x * 4 + threadIdx.x * 4;
    int y = blockIdx.y * blockDim.y * 4 + threadIdx.y * 4;

    for (int i = 0; i < 4; ++i) {
        FETCH_FLOA4(src_transpose[i][0]) = FETCH_FLOA4(in[(y + i) * cols + x]);
    }
    
    // 做转置, 每一行变为一列
    for (int i = 0; i < 4; ++i) {
        FETCH_FLOA4(dst_transpose[i][0]) = make_float4(src_transpose[0][i], src_transpose[1][i], src_transpose[2][i], src_transpose[3][i]);
    }
    
    // 注意写回的时候，是否合并访寸了
    for (int i = 0; i < 4; ++i) {
        FETCH_FLOA4(out[(x + i) * rows + y]) = FETCH_FLOA4(dst_transpose[i][0]);
    }
}


// CPU转置：逐行读取，逐列写入
void transpose_cpu(float* out, const float* in, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            out[j * rows + i] = in[i * cols + j];  // 行优先→列优先
        }
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
    dim3 grid(((cols >> 2) + block.x - 1) / block.x, 
             ((rows >> 2) + block.y - 1) / block.y);
    
    // 创建CUDA事件用于计时
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // 记录开始时间
    cudaEventRecord(start);
    
    constexpr int Y_NUM_PER_THREAD = 8 * 4;
    constexpr int X_NUM_PER_THREAD = 32 * 4;

    transpose_gpu<Y_NUM_PER_THREAD, X_NUM_PER_THREAD><<<grid, block>>>(d_out, d_in, rows, cols);
    
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