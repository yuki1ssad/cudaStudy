#include <cuda_runtime.h>
#include <iostream>
#include "00common.h"


__global__ void reduceUnroll(int *g_idata, int *g_odata, unsigned int n)
{
    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    // convert global data pointer to the local pointer of this block
    int *idata = g_idata + blockIdx.x * blockDim.x * 2;

    // unroll 2 blocks
    if (idx + blockDim.x < n)
    {
        g_idata[idx] += g_idata[idx + blockDim.x];
    }
    __syncthreads();

    //in-place reduction in global memory
    for(int stride = blockDim.x / 2; stride > 0; stride /= 2)
    {
        if(tid < stride)
        {
            idata[tid] += idata[tid + stride];
        }
        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0)
    {
        g_odata[blockIdx.x] = idata[0];
    }
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

    // initialiation
    int size = 1 << 24; // total num of elements
    std::cout << "   with arry size: " << size <<std::endl;

    // execution configuration
    int blocksize = 512;

    dim3 block (blocksize, 1);
    dim3 grid ((size + block.x - 1) / block.x, 1);
    std::cout << "   grid: " << grid.x  << "   block: " << block.x << std::endl;

    // allocate host memory
    size_t bytes = size * sizeof(int);
    int *h_idata = new int[size];
    int *h_odata = new int[grid.x];
    int *tmp = new int[size];

    // inilization the array
    for (int i = 0; i <size; ++i)
    {
        h_idata[i] = int(rand() % 0xff);
    }

    memcpy(tmp, h_idata, bytes);
    double iStart, iElaps;
    int gpu_sum = 0;

    // allocate device memory
    int *d_idata = nullptr;
    int *d_odata = nullptr;
    cudaMalloc((void **) &d_idata, bytes);
    cudaMalloc((void **) &d_odata, grid.x * sizeof(int));

    cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToHost);
    iStart = GetCPUSecond();
    reduceUnroll<<<grid, block>>>(d_idata, d_odata, size);
    cudaDeviceSynchronize();
    iElaps = GetCPUSecond() - iStart;
    cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
    gpu_sum = 0;

    for (int i = 0; i <grid.x; ++i)
    {
        gpu_sum += h_odata[i];
    }

    std::cout << "   gpu Unroll elapse: " << iElaps << "\t sec, gpu_sum: " << gpu_sum << std::endl;

    std::cout << "cpu start computing" << std::endl;
    double cpuStart = GetCPUSecond();
    int cpu_sum = 0;
    for (int i = 0; i <size; ++i) {
        cpu_sum += h_idata[i];
    }
    double cpuElaps = GetCPUSecond() - cpuStart;
    std::cout << "   cpu elapse: " << cpuElaps << "\t sec, cpu_sum: " << cpu_sum << std::endl;



    delete[] h_idata;
    delete[] h_odata;
    delete[] tmp;

    cudaFree (d_idata);
    cudaFree (d_odata);

    cudaDeviceReset();
    return 0;
}