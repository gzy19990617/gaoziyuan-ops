#include <bits/stdc++.h>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <time.h>
#include <sys/time.h>

#define THREAD_PER_BLOCK 256

// v2:避免线程分化，一个warp是32个线程，同一个warp中的不同分支失去了并发行。
// 不同warp之间又保留了并发性，会存在不一致，容易导致死锁

// a = (cond ? x[i]:0.f) 这种三元表达符号不会导致分支
// 让每轮迭代的前一半线程负责运算，组成一个warp

__global__ void reduce1(float* d_a, float* d_out) {
    __shared__ float s_a[THREAD_PER_BLOCK];

    // 搬运数据到共享内存中，每个线程搬运一个元素
    float* input_begein = d_a + blockIdx.x * blockDim.x;
    s_a[threadIdx.x] = input_begein[threadIdx.x];
    // 搬运完需要进行同步
    __syncthreads();

    for (int i = 1; i < blockDim.x; i *= 2) {
        if (threadIdx.x < blockDim.x / (2 * i)) {
            // 第一轮：0号线程负责，0+1. 1号线程负责，1+2. 以此类推
            // 第二轮：0号线程负责，0+2. 1号线程负责，4+6. 以此类推
            int index = threadIdx.x * (2 * i);
            s_a[index] += s_a[index + i];
        }
        // 每一次要等这一轮计算完
        __syncthreads();
    }
    //最终每个block把计算结果放在第一个索引的位置
    if (threadIdx.x == 0) {
        d_out[blockIdx.x] = input_begein[0];
    }
}

// __global__ void reduce1(float* d_a, float* d_out) {
//     __shared__ float s_a[THREAD_PER_BLOCK];
    
//     int tid = threadIdx.x;
//     int global_tid = blockIdx.x * blockDim.x + tid;

//     // 搬运数据到共享内存中，每个线程搬运一个元素
//     s_a[tid] = d_a[global_tid];

//     // 搬运完需要进行同步
//     __syncthreads();

//     for (int i = 1; i < blockDim.x; i *= 2) {
//         if (tid % (2 * i) == 0) {
//             d_a[global_tid] += d_a[global_tid + i];
//         }
//         // 每一次要等这一轮计算完
//         __syncthreads();
//     }
//     //最终每个block把计算结果放在第一个索引的位置
//     if (tid == 0) {
//         d_out[blockIdx.x] = d_a[global_tid];
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
    

    reduce1<<<Grid, Block>>>(d_a, d_out);

    cudaMemcpy(out, d_out, block_num*sizeof(float),cudaMemcpyDeviceToHost);

     if(check(out,res,block_num))printf("the ans is right\n");
    else{
        printf("the ans is wrong\n");
        for(int i=0;i<block_num;i++){
            printf("%lf ",out[i]);
        }
        printf("\n");
    }

    cudaFree(d_a);
    cudaFree(d_out);

    return 0;
}