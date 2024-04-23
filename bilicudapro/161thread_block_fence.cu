#include <cuda_runtime.h>
#include <iostream>

#include "00common.h"

__global__ void thread_block_fence()
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    __shared__ float shared;
    shared = 0.0;
    if ((id / 32) == 0 && id == 0)  // 第一个线程束的第一个线程
    {
        shared = 5.0;
    }
    else if ((id / 32) != 0 && id == 32)    // // 第二个线程束的第一个线程
    {
        shared = 6.0;
    }
    __threadfence_block();
    printf("access local shared in thread_fence, \tshared=%.2f, \tblockIdx=%d, \tthreadIdx=%d, \tthreadId=%d\n", shared, blockIdx.x, threadIdx.x, id);
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
    dim3 block (32);
    dim3 grid (2);

    thread_block_fence<<<grid, block>>>();

    cudaDeviceSynchronize();
    cudaDeviceReset();
    return 0;
}