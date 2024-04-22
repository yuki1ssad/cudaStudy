#include <cuda_runtime.h>
#include <iostream>

#include "00common.h"

__global__ void pageLockedMemory(float *input)
{
    printf("GPU pageLocked memory:%.2f\n", *input);
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
    dim3 block (1);
    dim3 grid (1);

    float *h_PinnedMem = nullptr;

    ErrorCheck(cudaMallocHost((float**)&h_PinnedMem, sizeof(float)), __FILE__, __LINE__);

    *h_PinnedMem = 4.8;

    pageLockedMemory<<<grid, block>>>(h_PinnedMem);
    cudaDeviceSynchronize();
    std::cout << "CPU pageLocked memory: " << *h_PinnedMem <<std::endl;


    cudaFree (h_PinnedMem);
    cudaDeviceReset();
    return 0;
}