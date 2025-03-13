#include <bits/stdc++.h>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <time.h>
#include <sys/time.h>

#define THREAD_PER_BLOCK 256

// v7:每个线程做更多的事情, block数量更多一些

// volatile 参考这个https://blog.csdn.net/seamanj/article/details/78453897
__device__ void warpReduce(volatile float* s_data, int tid)  // 此处不加volatile会有diff
{
    s_data[tid] += s_data[tid + 32];
    s_data[tid] += s_data[tid + 16];
    s_data[tid] += s_data[tid + 8];
    s_data[tid] += s_data[tid + 4];
    s_data[tid] += s_data[tid + 2];
    s_data[tid] += s_data[tid + 1];
}

template<unsigned int NUM_PER_BLOCK>
__global__ void reduce1(float* d_a, float* d_out) {
    
    __shared__ float s_a[THREAD_PER_BLOCK]; // 此处不加volatile会有diff

    int tid = threadIdx.x;
    float* input_begein = d_a + blockIdx.x * NUM_PER_BLOCK;
    s_a[tid] = 0;
    // 每个线程处理这么多数
    for (int i = 0 ;i < NUM_PER_BLOCK / THREAD_PER_BLOCK; i++) {
        s_a[tid] += input_begein[tid + i* THREAD_PER_BLOCK];
    }
    __syncthreads();

#pragma unroll
    for (int i = THREAD_PER_BLOCK / 2; i > 32; i /= 2) { // i > 32时才执行
        if (threadIdx.x < i) {
            s_a[threadIdx.x] += s_a[threadIdx.x + i];
            // warp与warp之间没有办法同步，但warp内同线程是可以同步的，所以后面几次迭代不需要同步了
            __syncthreads();
        }
    }
    if (tid < 32) {
        warpReduce(s_a, tid);

    }

    if (threadIdx.x == 0) {
        d_out[blockIdx.x] = s_a[0];
    }
}

// __global__ void reduce1(float* d_a, float* d_out) {
//     __shared__ float s_a[THREAD_PER_BLOCK];

//     // 搬运数据到共享内存中，每个线程搬运一个元素
//     int global_id = blockDim.x * blockIdx.x + threadIdx.x;
//     s_a[threadIdx.x] = d_a[global_id];
//     // 搬运完需要进行同步
//     __syncthreads();

//     for (int i = blockDim.x / 2; i > 0; i =/2) {
//         if (threadIdx.x < i) {
//             s_a[threadIdx.x] += s_a[threadIdx.x + i];
//         }
//         __syncthreads();
//     }
//     if (threadIdx.x == 0) {
//         d_out[blockIdx.x] = d_a[global_id];
//     }
// }


bool check(float *out,float *res,int n){
    for(int i=0;i<n;i++){
        if (abs(out[i] - res[i]) > 0.005) {
            return false;
        }
    }
    return true;
}

int main() {
    float milliseconds = 0;
    printf("hello \n");

    const int N=32*1024*1024;
    cudaSetDevice(0);

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);


    float *a=(float *)malloc(N*sizeof(float));
    float *d_a;
    cudaMalloc((void **)&d_a,N*sizeof(float));

    constexpr int block_num = 1024;
    constexpr int num_per_block = N / block_num;

    // int block_num = N / THREAD_PER_BLOCK / 2;

    float *out = (float *)malloc((block_num * sizeof(float)));
    float *d_out;
    cudaMalloc((void **)&d_out, block_num * sizeof(float));

    float *res=(float *)malloc((N/THREAD_PER_BLOCK)*sizeof(float));

    for(int i=0;i<N;i++){
        a[i]=1.f;
    }

    for(int i=0;i<block_num;i++){
        float cur=0;
        for(int j=0;j<num_per_block;j++){
            cur+=a[i*num_per_block+j];
        }
        res[i]=cur;
    }

    cudaMemcpy(d_a, a, N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 Grid(block_num);
    dim3 Block(THREAD_PER_BLOCK);
    // for (int i = 0; i < 10; i ++) {

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);


        reduce1<num_per_block><<<Grid, Block>>>(d_a, d_out);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);

        printf("reduce latency = %f ms\n", milliseconds);


    // }
    


    cudaMemcpy(out, d_out, block_num*sizeof(float),cudaMemcpyDeviceToHost);

     if(check(out,res,block_num))printf("the ans is right\n");
    else{
        printf("the ans is wrong\n");
        for(int i=0;i<block_num;i++){
            printf("%lf ",out[i]);
        }
        printf("\n");
        printf("the ans is wrong\n");
        for(int i=0;i<block_num;i++){
            printf("%lf ",res[i]);
        }
        printf("\n");
    }

    cudaFree(d_a);
    cudaFree(d_out);

    return 0;
}