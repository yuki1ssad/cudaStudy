#include "matrix_multiply.h"
#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cmath>

// 辅助函数，用于初始化矩阵和验证结果
void initializeMatrix(float* matrix, int size) {
    for (int i = 0; i < size * size; ++i) {
        matrix[i] = static_cast<float>(i);
    }
}

bool areMatricesEqual(float* matrix1, float* matrix2, int size, float tolerance = 1e-5) {
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            if (std::fabs(matrix1[i * size + j] - matrix2[i * size + j]) > tolerance) {
                return false;
            }
        }
    }
    return true;
}

// 测试CPU矩阵乘法
TEST(MatrixMultiplyTest, CPU) {
    const int size = 4;
    float A[size * size];
    float B[size * size];
    float C[size * size] = {0};
    float C_expected[size * size] = {0};

    initializeMatrix(A, size);
    initializeMatrix(B, size);
    matrixMultiplyCPU(A, B, C, size);
    matrixMultiplyCPU(A, B, C_expected, size); // 使用CPU版本作为预期结果

    EXPECT_TRUE(areMatricesEqual(C, C_expected, size));

    // std::cout << "CPU Matrix Multiply C:" << std::endl;
    // for (int i = 0; i < size; ++i) {
    //     for (int j = 0; j < size; ++j) {
    //         std::cout << "\t" << C[i * size + j] << " ";
    //     }
    //     std::cout << std::endl;
    // }

    // std::cout << "CPU Matrix Multiply C_expected:" << std::endl;
    // for (int i = 0; i < size; ++i) {
    //     for (int j = 0; j < size; ++j) {
    //         std::cout << "\t" << C_expected[i * size + j] << " ";
    //     }
    //     std::cout << std::endl;
    // }
}

// 测试GPU矩阵乘法
TEST(MatrixMultiplyTest, GPU) {
    const int size = 4;
    float *A, *B, *C, *C_expected;
    float host_A[size * size];
    float host_B[size * size];
    float host_C[size * size] = {0};
    float host_C_expected[size * size] = {0};

    initializeMatrix(host_A, size);
    initializeMatrix(host_B, size);

    // 分配设备内存
    cudaMalloc(&A, size * size * sizeof(float));
    cudaMalloc(&B, size * size * sizeof(float));
    cudaMalloc(&C, size * size * sizeof(float));

    // 将数据从主机复制到设备
    cudaMemcpy(A, host_A, size * size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B, host_B, size * size * sizeof(float), cudaMemcpyHostToDevice);

    // 执行GPU矩阵乘法
    matrixMultiplyGPU(A, B, C, size);

    // 将结果从设备复制回主机
    cudaMemcpy(host_C, C, size * size * sizeof(float), cudaMemcpyDeviceToHost);

    // 使用CPU版本验证GPU结果
    matrixMultiplyCPU(host_A, host_B, host_C_expected, size);

    // 验证结果是否一致
    EXPECT_TRUE(areMatricesEqual(host_C, host_C_expected, size));

    // 释放设备内存
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}