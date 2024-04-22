#include <cuda_runtime.h>
#include <iostream>

#include "00common.h"

__constant__ float factor;

__global__ void constantMemory()
{
    printf("Get constant memory:%.2f\n", factor);
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
    if (error != cudaSuccess || nDeviceNumber == 0)
    {
        std::cout << "Fail to set GPU 0 for computing" << std::endl;
        return -1;
    }
    else 
    {
        std::cout << "Set GPU 0 for computing" << std::endl;
    }

    // calculate on GPU
    dim3 block (8, 1);
    dim3 grid (1, 1);

    float h_factor = 2.3;
    ErrorCheck(cudaMemcpyToSymbol(factor, &h_factor, sizeof(float), 0, cudaMemcpyHostToDevice), __FILE__, __LINE__);
    constantMemory<<<grid, block>>>();
    cudaDeviceSynchronize();

    cudaDeviceReset();
    return 0;
}