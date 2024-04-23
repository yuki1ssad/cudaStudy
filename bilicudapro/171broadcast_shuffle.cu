#include <cuda_runtime.h>
#include <iostream>

#include "00common.h"

__global__ void shfl_broadcast(int *in, int *out, int const srcLane)
{
    int value = in[threadIdx.x];
    value = __shfl(value, srcLane, 32); // 将第 srcLane 条线程的值传递给其他线程的 value 变量
    out[threadIdx.x] = value;
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

    int nElem = 32;

    int *in = nullptr;
    int *out = nullptr;

    ErrorCheck(cudaHostAlloc((void**)&in, sizeof(int) * nElem, cudaHostAllocDefault), __FILE__, __LINE__);
    ErrorCheck(cudaHostAlloc((void**)&out, sizeof(int) * nElem, cudaHostAllocDefault), __FILE__, __LINE__);

    for (int i = 0; i < nElem; ++i)
    {
        in[i] = i;
    }

    // calculate on gpu
    dim3 block (nElem);
    dim3 grid (1);

    shfl_broadcast<<<grid, block>>>(in, out, 3);
    cudaDeviceSynchronize();

    for (int i = 0; i < nElem; ++i)
    {
        std::cout << "out element is, id=" << i << "\tvalue=" << out[i] << std::endl;
    }

    cudaFreeHost(in);
    cudaFreeHost(out);
    cudaDeviceReset();
    return 0;
}
