// v0:global_memory最基本写法

#include <bits/stdc++.h>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <time.h>
#include <sys/time.h>

#define THREAD_PER_BLOCK 256

// 设计算法的时候按照block来设计，但是写程序的时候是按照thread来写的； trition是按照block来设计，block来写的
// 索引容易乱套，最好为每一个block设计一个更进一步的索引
// 绘制一个清晰的算法图很有必要

// 0.760288 ms

__global__ void reduce0(float* d_a, float* d_out) {
    // printf("haha \n");
    // 每个block里面的第一个索引
    float* input_begein = d_a + blockIdx.x * blockDim.x;
    for (int i = 1; i < blockDim.x; i *= 2) {
        if (threadIdx.x % (2 * i) == 0) {
            input_begein[threadIdx.x] = input_begein[threadIdx.x] + input_begein[threadIdx.x + i];
        }
        // 每一次要等这一轮计算完
        __syncthreads();
    }
    //最终每个block把计算结果放在第一个索引的位置
    if (threadIdx.x == 0) {
        d_out[blockIdx.x] = input_begein[0];
    }

    // 思考过程, 需要先把流程图给画出来
    // if (threadIdx.x == 0 or 2 or 4 or 6) {
    //     input_begein[threadIdx] = input_begein[threadIdx] + input_begein[threadIdx + 1];
    // }
    // if (threadIdx.x == 0 or 4) {
    //     input_begein[threadIdx] = input_begein[threadIdx] + input_begein[threadIdx + 2];
    // }
    // if (threadIdx.x == 0) {
    //     input_begein[threadIdx] = input_begein[threadIdx] + input_begein[threadIdx + 4];
    // }
}

// 另一种索引方式, 注意是全局索引
// baseline, 使用global memory, 但把输入数据给修改了，这样并不好
__global__ void reduce1(float* d_a, float* d_out) {
    int tid = threadIdx.x;
    // 线程的全局索引
    int global_tid  = tid + blockIdx.x * blockDim.x;
    for (int i = 1; i < blockDim.x; i *= 2) {
        if (tid % (2 * i) == 0) {
            d_a[global_tid] += d_a[global_tid + i];
        }
        // 每一次要等这一轮计算完
        __syncthreads();
    }
    //最终每个block把计算结果放在第一个索引的位置
    if (tid == 0) {
        d_out[blockIdx.x] = d_a[global_tid];
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
    float *a=(float *)malloc(N*sizeof(float));
    float *d_a;
    cudaMalloc((void **)&d_a,N*sizeof(float));


    int block_num = N / THREAD_PER_BLOCK;

    float *out = (float *)malloc((block_num * sizeof(float)));
    float *d_out;
    cudaMalloc((void **)&d_out, block_num * sizeof(float));

    float *res=(float *)malloc((N/THREAD_PER_BLOCK)*sizeof(float));

    for(int i=0;i<N;i++){
        a[i]=1.2;
    }

    for(int i=0;i<block_num;i++){
        float cur=0;
        for(int j=0;j<THREAD_PER_BLOCK;j++){
            cur+=a[i*THREAD_PER_BLOCK+j];
        }
        res[i]=cur;
    }

    cudaMemcpy(d_a, a, N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 Grid(block_num);
    dim3 Block(THREAD_PER_BLOCK);
    

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    reduce1<<<Grid, Block>>>(d_a, d_out);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);


    cudaMemcpy(out, d_out, block_num*sizeof(float),cudaMemcpyDeviceToHost);

     if(check(out,res,block_num))printf("the ans is right\n");
    else{
        printf("the ans is wrong\n");
        for(int i=0;i<block_num;i++){
            printf("%lf ",out[i]);
        }
        printf("\n");
    }
    printf("reduce_v0 latency = %f ms\n", milliseconds);

    cudaFree(d_a);
    cudaFree(d_out);

    return 0;
}