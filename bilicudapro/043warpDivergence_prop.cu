#include <cuda_runtime.h>
#include "00common.h"

__global__ void mathKernel(float *c)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float ia, ib;
    ia = ib = 0.0f;

    // 分支特点： warp 分支只会发生在同一 warp 内
    // 如果代码中的条件判断值与线程 id 关联，则以线程束为基本单元访问数
    if ((tid / warpSize) % 2 == 0)
    {
        ia = 100.0f;
    }
    else
    {
        ib = 200.0f;
    }
    
    c[tid] = ia + ib;
}

int main(int argc, char **argv)
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

    // set up data size
    int size = 64;
    int blocksize = 64;

    std::cout << "Data size : " << size << std::endl;

    dim3 block (blocksize, 1);
    dim3 grid ((size + block.x - 1) / block.x);

    float *d_C;
    size_t nBytes = size * sizeof(float);
    error = ErrorCheck(cudaMalloc((float**)&d_C, nBytes), __FILE__, __LINE__);
    if (error != cudaSuccess)
    {
        std::cout << "Fail to set GPU 0 for computing" << std::endl;
        return -1;
    }

    mathKernel<<<grid, block>>>(d_C);
    cudaDeviceSynchronize();

    cudaFree(d_C);
    cudaDeviceReset();
    return 0;
}