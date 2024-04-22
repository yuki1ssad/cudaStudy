#include <cuda_runtime.h>
#include <iostream>

#include "00common.h"

__managed__ float y = 9.0;

__global__ void unifiedMemory(float *A)
{
    *A += y;
    printf("GPU unified memory:%.2f\n", *A);
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

    // check whether to support unified memory
    int supportManagedMemory = 0;
    ErrorCheck(cudaDeviceGetAttribute(&supportManagedMemory, cudaDevAttrManagedMemory, dev), __FILE__, __LINE__);

    if (0 == supportManagedMemory)
    {
        std::cout << "Allocate managed memory is not supported" << std::endl;
        return -1;
    }
    std::cout << "Unified managed memory is supported" << std::endl;
    
    // calculate on gpu
    dim3 block (1);
    dim3 grid (1);

    float *unified_mem = nullptr;

    ErrorCheck(cudaMallocManaged((void**)&unified_mem, sizeof(float), cudaMemAttachGlobal), __FILE__, __LINE__);

    *unified_mem = 5.7;

    unifiedMemory<<<grid, block>>>(unified_mem);
    cudaDeviceSynchronize();
    std::cout << "CPU unified memory: " << *unified_mem <<std::endl;


    cudaFree (unified_mem);
    cudaDeviceReset();
    return 0;
}