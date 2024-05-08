#include "matrix_multiply.h"
#include <iostream>

void matrixMultiplyCPU(float* A, float* B, float* C, int size) {
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            for (int k = 0; k < size; ++k) {
                C[i * size + j] += A[i * size + k] * B[k * size + j];
            }
        }
    }
}