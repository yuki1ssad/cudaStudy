#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <cstring>

#include "00common.h"

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

__global__ void sumArraysOnGPU(float* A, float* B, float* C, const int N)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i < N)
    {
        C[i] = A[i] + B[i];
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

    // set up data size of vectors
    int nElem = 1 << 14;

    // malloc host memory
    
    float *h_A, *h_B, *gpuRef;
    h_A = new float[nElem];
    h_B = new float[nElem];
    gpuRef = new float[nElem];

    if(h_A && h_B && gpuRef)
    {
        std::cout << "Allocate memory successfully!" <<std::endl;
    }
    else
    {
        std::cout << "Fail to allocate memory" <<std::endl;
        return -1;
    }

    // initialize data at host side
    initialData(h_A, nElem);
    initialData(h_B, nElem);
    std::memset(gpuRef, 0., nElem * sizeof(float));

    // allocate GPU memory
    float *d_A, *d_B, *d_C;
    size_t nBytes = nElem * sizeof(float);
    cudaMalloc((float**)&d_A, nBytes);
    cudaMalloc((float**)&d_B, nBytes);
    cudaMalloc((float**)&d_C, nBytes);

    if(!d_A || !d_B || !d_C)
    {
        std::cout << "Fail to allocate memory on GPU" <<std::endl;
        delete[] h_A;
        delete[] h_B;
        delete[] gpuRef;
    }
    else
    {
        std::cout << "Successfully allocate memory on GPU" <<std::endl;
    }

    // transfer data from host to device
    if (
        cudaSuccess == cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice) &&
        cudaSuccess == cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice) &&
        cudaSuccess == cudaMemcpy(d_C, gpuRef, nBytes, cudaMemcpyHostToDevice)
    )
    {
        std::cout << "Successfully copy data from CPU to GPU" <<std::endl;
    }
    else
    {
        std::cout << "Failed to copy data from CPU to GPU" <<std::endl;
    }

    // calculate on GPU
    dim3 block (32);
    dim3 grid (nElem / 32);
    std::cout << "Execution congigure: <<<" << nElem / 32 << ", " << 32 << ">>>" << std::endl;
    double dTime_Begin = GetCPUSecond();
    sumArraysOnGPU<<<grid, block>>>(d_A, d_B, d_C, nElem);
    cudaDeviceSynchronize();
    double dTime_End = GetCPUSecond();
    cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost);

    for(int i = 0; i < 200; ++i)
    {
        std::cout << "idx=" << i + 1 <<"\tmatrex_A: " << h_A[i] << "\tmatrix_B: " << h_B[i] << "\tresult=" << gpuRef[i] << std::endl;
    }
    std::cout << "Element Size: " << nElem << " Matrix add time Elapse is : " << std::setprecision(6) << dTime_End - dTime_Begin << std::endl;


    delete[] h_A;
    delete[] h_B;
    delete[] gpuRef;
    return 0;
}