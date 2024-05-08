#include "matrix_multiply.h"
#include <cuda_runtime.h>

__global__ void matrixMultiply(float* A, float* B, float* C, int size) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row_o = blockIdx.y * blockDim.y + ty;
    int col_o = blockIdx.x * blockDim.x + tx;

    // 确保线程处理的矩阵元素在矩阵的边界内
    if (row_o < size && col_o < size) {
        float result = 0.0f;
        for (int i = 0; i < size; ++i) {
            result += A[row_o * size + i] * B[i * size + col_o];
        }
        C[row_o * size + col_o] = result;
    }
}

void matrixMultiplyGPU(float* A, float* B, float* C, int size) {
    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, size * size * sizeof(float));
    cudaMalloc((void**)&d_B, size * size * sizeof(float));
    cudaMalloc((void**)&d_C, size * size * sizeof(float));

    // Transfer data from host to device
    cudaMemcpy(d_A, A, size * size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size * size * sizeof(float), cudaMemcpyHostToDevice);

    // Invoke the kernel
    dim3 dimGrid(ceil(size / 16.0), ceil(size / 16.0), 1);
    dim3 dimBlock(16, 16, 1);
    matrixMultiply<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, size);

    // Transfer data back to host
    cudaMemcpy(C, d_C, size * size * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}