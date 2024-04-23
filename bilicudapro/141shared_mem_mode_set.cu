#include <cuda_runtime.h>
#include <iostream>

#include "00common.h"

extern __shared__ int dynamic_arry[];

__global__ void dynamicSharedMem()
{
    dynamic_arry[threadIdx.x] = threadIdx.x;
    printf("access dynamic_array in kernel, dynamic_array[%d]=%d\n", threadIdx.x, dynamic_arry[threadIdx.x]);
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

    // get current shared memory mode
    cudaSharedMemConfig sharedMemCfg;
    ErrorCheck(cudaDeviceGetSharedMemConfig(&sharedMemCfg), __FILE__, __LINE__);
    std::cout << "current shared memory mode: " << sharedMemCfg << std::endl;

    if (cudaSharedMemBankSizeEightByte != sharedMemCfg)
    {
        sharedMemCfg = cudaSharedMemBankSizeEightByte;
        ErrorCheck(cudaDeviceSetSharedMemConfig(sharedMemCfg), __FILE__, __LINE__);
    }
    else if (cudaSharedMemBankSizeFourByte != sharedMemCfg)
    {
        sharedMemCfg = cudaSharedMemBankSizeFourByte;
        ErrorCheck(cudaDeviceSetSharedMemConfig(sharedMemCfg), __FILE__, __LINE__);
    }

    std::cout << "after modified, current shared memory mode: " << sharedMemCfg << std::endl;

    cudaDeviceReset();
    return 0;
}