#include <cuda_runtime.h>
#include <iostream>

#include "00common.h"

__global__ void thread_barrier()
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    __shared__ float shared;
    shared = 0.0;
    if ((id / 32) == 0)
    {
        __syncthreads();    // 避免在这样的分支里面用，因为分支对线程块中的所有线程结果不一致
        shared = 5.0;
    }
    else
    {
        // while(shared == 0.0)
        // {

        // }
    }
    printf("access local shared in thread_barrier, shared=%.2f, blockIdx=%d, threadIdx=%d, threadId=%d\n", shared, blockIdx.x, threadIdx.x, id);
}

int main(int argc, char** argv)
{
    // get GPU device count
    int nDeviceNumber = 0;
    cudaError_t error = ErrorCheck(cudaGetDeviceCount(&nDeviceNumber), __FILE__, __LINE__);
    if (error != cudaSuccess || nDeviceNumber == 0)
    {
        std::cout << "No CUDA campatable GPU found" << std::endl;
        return -1;
    }

    // set up device
    int dev = 0;
    error = ErrorCheck(cudaSetDevice(dev), __FILE__, __LINE__);
    if (error != cudaSuccess)
    {
        std::cout << "Fail to set GPU 0 for computing" << std::endl;
        return -1;
    }
    else 
    {
        std::cout << "Set GPU 0 for computing" << std::endl;
    }

    // calculate on gpu
    dim3 block (64);
    dim3 grid (1);

    thread_barrier<<<grid, block>>>();

    cudaDeviceSynchronize();
    cudaDeviceReset();
    return 0;
}