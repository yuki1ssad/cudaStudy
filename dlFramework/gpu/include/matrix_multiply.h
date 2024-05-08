#ifndef MATRIX_MULTIPLY_H
#define MATRIX_MULTIPLY_H

void matrixMultiplyCPU(float* A, float* B, float* C, int size);
void matrixMultiplyGPU(float* A, float* B, float* C, int size); // 确保这里声明了GPU版本

#endif // MATRIX_MULTIPLY_H