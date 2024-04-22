#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <cstring>

#include "00common.h"

void printMatrix(int* C, const int nx, const int ny)
{
    std::cout << std::endl;
    for (int iy = 0; iy < ny; ++iy)
    {
        for(int ix = 0; ix < nx; ++ix)
        {
            std::cout << "\t" << C[iy * nx + ix];
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
    return;
}

void initialData(float* ip, int size)
{
    time_t t;
    srand((unsigned) time(&t));
    std::cout << "Matrix is: ";
    for (int i = 0; i < size; ++i)
    {
        ip[i] = static_cast<float>(rand() & 0xff) / 10.0;
        std::cout << std::fixed << std::setprecision(2) << ip[i] << " ";
    }
    std::cout <<std::endl;
    return;
}

__global__ void printThreadIndex(int *A, const int nx, const int ny)
{
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idx = iy * nx + ix;
    printf("thread_id (%d, %d) \tblock_id (%d, %d) \tcoordinate (%d, %d) \tglobal index %2d \tival %2d\n", threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y, ix, iy, idx, A[idx]);
    
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

    // set matrix dimension
    int nx = 8;
    int ny = 6;
    int nxy = nx * ny;
    int nBytes = nxy * sizeof(int);

    // malloc host memory

    int *h_A;
    h_A = new int[nxy];

    // initialize data at host side
    for(int i = 0; i < nxy; ++i)
    {
        h_A[i] = i;
    }
    printMatrix(h_A, nx, ny);

    // allocate GPU memory
    int *d_MatA;
    cudaMalloc((void**)&d_MatA, nBytes);

    // transfer data from host to device
    if (
        cudaSuccess == cudaMemcpy(d_MatA, h_A, nBytes, cudaMemcpyHostToDevice)
    )
    {
        std::cout << "Successfully copy data from CPU to GPU" <<std::endl;
    }
    else
    {
        std::cout << "Failed to copy data from CPU to GPU" <<std::endl;
    }

    // calculate on GPU
    dim3 block (4, 2);
    dim3 grid ((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);
    
    double dTime_Begin = GetCPUSecond();
    printThreadIndex<<<grid, block>>>(d_MatA, nx, ny);
    cudaDeviceSynchronize();
    double dTime_End = GetCPUSecond();

    std::cout << "Element Size: " << nxy << " Matrix add time Elapse is : " << std::setprecision(6) << dTime_End - dTime_Begin << std::endl;

    // cudaFree (d_MatA);
    delete[] h_A;

    cudaDeviceReset();
    return 0;
}