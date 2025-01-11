#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>




typedef float FLOAT;


__global__ void vec_add(FLOAT* x, FLOAT* y, FLOAT* z, int N) {

    // 1D
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        z[idx] = y[idx] + x[idx];
    }
}

void vec_add_cpu(FLOAT* x, FLOAT* y, FLOAT* z, int N) {
    for (int i =0; i < N; i++) {
        z[i] = y[i] + x[i];
    }
}

int main() {
    int N = 10000;
    int nbytes = N * sizeof(FLOAT);


    // 一维grid，每个block里面有256个线程
    int bs = 256;
    int s = ceil((N + bs - 1.0) / bs);
    dim3 grid(s);


    FLOAT *dx, *hx;
    FLOAT *dy, *hy;
    FLOAT *dz, *hz;

    cudaMalloc((void **)&dx, nbytes);
    cudaMalloc((void **)&dy, nbytes);
    cudaMalloc((void **)&dz, nbytes);

    hx = (FLOAT*) malloc(nbytes);
    hy = (FLOAT*) malloc(nbytes);
    hz = (FLOAT*) malloc(nbytes);

    for (int i =0;i<N;i++){
        hx[i] = 1;
        hy[i] = 1;
    }

    cudaMemcpy(dx, hx, nbytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dy, hy, nbytes, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    float milliseconds = 0;

    vec_add<<<grid, bs>>>(dx,dy,dz,N);

     cudaEventRecord(stop);
     cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);  

    cudaMemcpy(hz, dz, nbytes, cudaMemcpyDeviceToHost);

     /* CPU compute */
    FLOAT* hz_cpu_res = (FLOAT *) malloc(nbytes);
    vec_add_cpu(hx, hy, hz_cpu_res, N);

    for (int i = 0; i < N; ++i) {
        if (fabs(hz_cpu_res[i] - hz[i]) > 1e-6) {
            printf("Result verification failed at element index %d!\n", i);
        }
    }
    printf("Result right\n");
    cudaFree(dx);
    cudaFree(dy);
    cudaFree(dz);

    free(hx);
    free(hy);
    free(hz);
    free(hz_cpu_res);

    return 0;
}