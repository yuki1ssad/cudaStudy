#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <cstring>

#include "00common.h"

void initialData(float* ip, int size)
{
    time_t t;
    srand((unsigned) time(&t));
    // std::cout << "Matrix is: ";
    for (int i = 0; i < size; ++i)
    {
        ip[i] = static_cast<float>(rand() & 0xff) / 10.0;
        // std::cout << std::fixed << std::setprecision(2) << ip[i] << " ";
    }
    // std::cout <<std::endl;
    return;
}

__global__ void sumArraysOnGPU(float* A, float* B, float* C, const int N)
{
    int i = threadIdx.x;
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

    // get the supported priority on this deice
    int lowPriority = 0;
    int highPriority = 0;
    cudaDeviceGetStreamPriorityRange(&lowPriority, &highPriority);
    std::cout << "lowPriority: " << lowPriority <<"\thighPriority: " << highPriority << std::endl;

    // set up data size of vectors
    int nElem = 1 << 24;

    // malloc host pinned memory
    
    float *pinned_A, *pinned_B, *h_C;
    size_t nBytes = nElem * sizeof(float);
    ErrorCheck(cudaHostAlloc((void**)&pinned_A, nBytes, cudaHostAllocDefault), __FILE__, __LINE__);
    ErrorCheck(cudaHostAlloc((void**)&pinned_B, nBytes, cudaHostAllocDefault), __FILE__, __LINE__);
    h_C = new float[nElem];

    if(pinned_A && pinned_B && h_C)
    {
        std::cout << "Allocate memory successfully!" <<std::endl;
    }
    else
    {
        std::cout << "Fail to allocate memory" <<std::endl;
        return -1;
    }

    // initialize data at host side
    initialData(pinned_A, nElem);
    initialData(pinned_B, nElem);

    // allocate GPU memory
    float *d_A, *d_B, *d_C;
    cudaMalloc((float**)&d_A, nBytes);
    cudaMalloc((float**)&d_B, nBytes);
    cudaMalloc((float**)&d_C, nBytes);

    // transfer data from host to device
    cudaStream_t data_stream;
    cudaStreamCreate(&data_stream);

    cudaMemcpyAsync(d_A, pinned_A, nBytes, cudaMemcpyHostToDevice, data_stream);
    cudaEvent_t cp_evt_A;
    ErrorCheck(cudaEventCreate(&cp_evt_A), __FILE__, __LINE__);
    
    cudaMemcpyAsync(d_B, pinned_B, nBytes, cudaMemcpyHostToDevice, data_stream);
    cudaEvent_t cp_evt_B;
    ErrorCheck(cudaEventCreate(&cp_evt_B), __FILE__, __LINE__);
    
    cudaStreamSynchronize(data_stream);

    // calculate on GPU
    dim3 block (512);
    dim3 grid ((nElem + block.x - 1) / block.x, 1);

    cudaStream_t kernelStream;
    cudaStreamCreateWithPriority(&kernelStream, cudaStreamDefault, highPriority);

    sumArraysOnGPU<<<grid, block, 0, kernelStream>>>(d_A, d_B, d_C, nElem);
    cudaMemcpy(h_C, d_C, nBytes, cudaMemcpyDeviceToHost);

    for(int i = 0; i < 50; ++i)
    {
        std::cout << "idx=" << i + 1 <<"\tmatrex_A: " << pinned_A[i] << "\tmatrix_B: " << pinned_B[i] << "\tresult=" << h_C[i] << std::endl;
    }


    cudaFreeHost (pinned_A);
    cudaFreeHost (pinned_B);
    delete[] h_C;
    cudaFree (d_A);
    cudaFree (d_B);
    cudaFree (d_C);
    cudaStreamDestroy(data_stream);

    cudaEventDestroy(cp_evt_A);
    cudaEventDestroy(cp_evt_B);

    cudaDeviceReset();
    return 0;
}