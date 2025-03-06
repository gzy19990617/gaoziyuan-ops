#include <bits/stdc++.h>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <time.h>
#include <sys/time.h>

#define THREAD_PER_BLOCK 256
#define WARP_SIZE 32
// v8:使用shuffle

// Shuffle指令是一组针对warp的指令。
// Shuffle指令最重要的特性就是warp内的寄存器可以相互访问。
// 在没有shuffle指令的时候，各个线程在进行通信时只能通过shared memory来访问彼此的寄存器。
// 而采用了shuffle指令之后，warp内的线程可以直接对其他线程的寄存器进行访存。

template<unsigned int NUM_PER_BLOCK>
__global__ void reduce1(float* d_a, float* d_out) {
    // 对寄存器进行操作

    float sum = 0.f;
    int tid = threadIdx.x;
    float* input_begein = d_a + blockIdx.x * NUM_PER_BLOCK;
    for (int i = 0 ;i < NUM_PER_BLOCK / THREAD_PER_BLOCK; i++) {
        sum += input_begein[tid + i* THREAD_PER_BLOCK];
    }

    // __syncthreads();此时不需要同步了，针对一个warp进行规约，warp自带同步
    sum += __shfl_down_sync(0xffffffff, sum, 16);
    sum += __shfl_down_sync(0xffffffff, sum, 8);
    sum += __shfl_down_sync(0xffffffff, sum, 4);
    sum += __shfl_down_sync(0xffffffff, sum, 2);
    sum += __shfl_down_sync(0xffffffff, sum, 1);

    // 当前每个warp里面都有一个num，借助shared_memory进行规约,因为不同warp之间不能直接通信
    // 一个block里面最多有32个warp，每个warp里面有32个数，1024个线程，新的架构可能更多
    __shared__ float wrapLevelSums[32];
    const int laneId = threadIdx.x % WARP_SIZE; // 每个warp里面的线程id
    const int warpId = threadIdx.x / WARP_SIZE; // 每个warp的id

    if (laneId == 0) {
        wrapLevelSums[warpId] = sum;
    }
    __syncthreads();

    // 剩余的计算使用第一个Warp来做
    if (warpId == 0) {
        // shared_memory 到寄存器, 一个wrap里面后面的数需要为0，否则会出错
        sum = (laneId < blockDim.x / 32) ? wrapLevelSums[laneId] : 0;
        sum += __shfl_down_sync(0xffffffff, sum, 16);
        sum += __shfl_down_sync(0xffffffff, sum, 8);
        sum += __shfl_down_sync(0xffffffff, sum, 4);
        sum += __shfl_down_sync(0xffffffff, sum, 2);
        sum += __shfl_down_sync(0xffffffff, sum, 1);

    }
    if (threadIdx.x == 0) {
        d_out[blockIdx.x] = sum;
    }
}


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