#include "matrix_multiply.h"
#include <iostream>
#include <cuda_runtime.h>

int main() {
    const int size = 4; // 矩阵的大小
    float *h_A = new float[size * size];
    float *h_B = new float[size * size];
    float *h_C_cpu = new float[size * size];
    float *h_C_gpu = new float[size * size];

    // 初始化矩阵A和B
    for (int i = 0; i < size * size; ++i) {
        h_A[i] = static_cast<float>(i);
        h_B[i] = static_cast<float>(i);
    }

    // CPU矩阵乘法
    matrixMultiplyCPU(h_A, h_B, h_C_cpu, size);

    // GPU矩阵乘法
    matrixMultiplyGPU(h_A, h_B, h_C_gpu, size);

    // 打印矩阵
    std::cout << "Matrix A:" << std::endl;
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            std::cout << h_A[i * size + j] << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "Matrix B:" << std::endl;
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            std::cout << h_B[i * size + j] << " ";
        }
        std::cout << std::endl;
    }

    // 打印结果
    std::cout << "CPU Matrix Multiply Result:" << std::endl;
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            std::cout << h_C_cpu[i * size + j] << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "GPU Matrix Multiply Result:" << std::endl;
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            std::cout << h_C_gpu[i * size + j] << " ";
        }
        std::cout << std::endl;
    }

    // 释放分配的内存
    delete[] h_A;
    delete[] h_B;
    delete[] h_C_cpu;
    delete[] h_C_gpu;

    cudaDeviceReset(); // 重置CUDA设备

    return 0;
}