#include <omp.h>
#include <iostream>


void omp_matrix(float *hostA, float *hostB, float *hostC, int M, int K, int N){
    float tmp = 0;
    #pragma omp for private(tmp)
    for(int i = 0; i < M; i++){
        for(int j = 0; j < N; j++){
            tmp = 0;
            for(int k = 0; k < K; k++){
                tmp += hostA[i * K + k] * hostB[k * N + j];
            }
            hostC[i * N + j] = tmp;
        }
    }
}