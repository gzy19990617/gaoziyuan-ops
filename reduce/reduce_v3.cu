#include <bits/stdc++.h>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <time.h>
#include <sys/time.h>

#define THREAD_PER_BLOCK 256

// v3:避免banck conflict
// 0.303648 ms

// 为了提高内存读写带宽，共享内存被分割成了32个等大小的内存块，即bank。因为一个warp有32个线程，相当于一个线程对应一个内存bank

// 避免同一个warp的线程访问同一个bank，但如果是访问同一个bank中的同一位置，会产生广播，不会发生conflict


// bank 0 : 0 32 64 96...
// bank 1: 1 33 65 97...

// 只看一个block中的thread 0 :
// 第一次迭代，thread 0 访问 0，1 ，thread 16 访问 32，33， 0/32都在bank0，同一个warp访问同一块warp了

// 优化后，每次一个thread访问0和总数据的一半
// 第一次迭代：
// thread0访问：0，128
// thread8:8,136
// thread16:16,144
// 第二次迭代：
// thread0访问：0，64
// thread8:8,72
// thread16:16,60

__global__ void reduce1(float* d_a, float* d_out) {
    __shared__ float s_a[THREAD_PER_BLOCK];

    // 搬运数据到共享内存中，每个线程搬运一个元素
    float* input_begein = d_a + blockIdx.x * blockDim.x;
    s_a[threadIdx.x] = input_begein[threadIdx.x];
    // 搬运完需要进行同步
    __syncthreads();

    for (int i = blockDim.x / 2; i > 0; i /= 2) {
        if (threadIdx.x < i) {
            s_a[threadIdx.x] += s_a[threadIdx.x + i];
        }
        __syncthreads();
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

    printf("reduce latency = %f ms\n", milliseconds);

    cudaFree(d_a);
    cudaFree(d_out);

    return 0;
}