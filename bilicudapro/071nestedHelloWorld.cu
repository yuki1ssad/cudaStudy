#include <cuda_runtime.h>
#include <iostream>
#include "00common.h"

__global__ void nestedHelloWorld(int const iSize, int iDepth)
{
    int tid = threadIdx.x;
    printf("Recursion = %d: Hello World from thread %d block %d\n", iDepth, tid, blockIdx.x);

    if (iSize == 1) return;

    int nthreads = iSize >> 1;

    // thread 0 launches child grid recursively
    if (tid == 0 && nthreads > 0)
    {
        nestedHelloWorld<<<1, nthreads>>>(nthreads, ++iDepth);
        printf("-------->nested execution depth: %d\n", iDepth);
    }
}

int main(int argc, char **argv)
{
    int size = 8;
    int blocksize = 8;
    int igrid = 1;

    if (argc >1)
    {
        igrid = std::atoi(argv[1]);
        size = igrid * blocksize;
    }

    dim3 block (blocksize, 1);
    dim3 grid ((size + block.x - 1) / block.x, 1);

    std::cout <<"Execution Configuration: grid " << grid.x << ", block " << block.x << std::endl;

    nestedHelloWorld<<<grid, block>>>(size, 0);

    cudaDeviceReset();
    return 0;
}

/*
    nvcc -arch=sm_75 -rdc=true 071nestedHelloWorld.cu -o 071nestedHelloWorld.o && ./071nestedHelloWorld.o

        Execution Configuration: grid 1, block 8
        Recursion = 0: Hello World from thread 0 block 0
        Recursion = 0: Hello World from thread 1 block 0
        Recursion = 0: Hello World from thread 2 block 0
        Recursion = 0: Hello World from thread 3 block 0
        Recursion = 0: Hello World from thread 4 block 0
        Recursion = 0: Hello World from thread 5 block 0
        Recursion = 0: Hello World from thread 6 block 0
        Recursion = 0: Hello World from thread 7 block 0
        -------->nested execution depth: 1
        Recursion = 1: Hello World from thread 0 block 0
        Recursion = 1: Hello World from thread 1 block 0
        Recursion = 1: Hello World from thread 2 block 0
        Recursion = 1: Hello World from thread 3 block 0
        -------->nested execution depth: 2
        Recursion = 2: Hello World from thread 0 block 0
        Recursion = 2: Hello World from thread 1 block 0
        -------->nested execution depth: 3
        Recursion = 3: Hello World from thread 0 block 0
*/