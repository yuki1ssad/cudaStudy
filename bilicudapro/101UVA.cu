#include <cuda_runtime.h>
#include <iostream>

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
    if (error != cudaSuccess)
    {
        std::cout << "Fail to set GPU 0 for computing" << std::endl;
        return -1;
    }
    else 
    {
        std::cout << "Set GPU 0 for computing" << std::endl;
    }

    float *d_mem = nullptr;
    ErrorCheck(cudaMalloc((void**)&d_mem, sizeof(float)), __FILE__, __LINE__);

    cudaPointerAttributes pt_Attribute;
    ErrorCheck(cudaPointerGetAttributes(&pt_Attribute, d_mem), __FILE__, __LINE__);
    std::cout << "pointer Attribute:device=" << pt_Attribute.device << "\tdevicePointer=" << pt_Attribute.devicePointer << "\ttype=" <<pt_Attribute.type << std::endl;

    cudaFree (d_mem);
    cudaDeviceReset();
    return 0;
}