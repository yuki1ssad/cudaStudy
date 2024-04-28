#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <cstring>

#include "00common.h"

__global__ void infiniteKernel()
{
    while (true)
    {
        /* code */
    }
    
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

    
    int nElem = 16;

    // calculate on GPU
    dim3 block (nElem);
    dim3 grid (1);

    cudaStream_t kernelStream;
    cudaStreamCreate(&kernelStream);
    infiniteKernel<<<grid, block, 0, kernelStream>>>();

    cudaEvent_t kernelEvent;
    ErrorCheck(cudaEventCreateWithFlags(&kernelEvent, cudaEventBlockingSync), __FILE__, __LINE__);
    ErrorCheck(cudaEventRecord(kernelEvent, kernelStream), __FILE__, __LINE__);


    // wait for data copy to complete
    cudaEventSynchronize(kernelEvent);
    std::cout << "Event kernelEvent is finished" << std::endl;


    cudaStreamDestroy(kernelStream);
    cudaEventDestroy(kernelEvent);

    cudaDeviceReset();
    return 0;
}