#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <cstring>

#include "00common.h"

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

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);

    if (deviceProp.concurrentKernels)
    {
        std::cout << "concurrent kernel is supported on this GPU, begin to execute kernel_1" << std::endl;
    }
    else
    {
        std::cout << "concurrent kernel is Not supported on this GPU." << std::endl; 
    }
    
    cudaDeviceReset();
    return 0;
}