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

void data_cp_callback(cudaStream_t stream, cudaError_t status, void* userData)
{
    printf("data copy callback is invoked, datasize: %d\n", *((int*)userData));
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

    int nElem = 1 << 12;

    size_t nBytes = nElem * sizeof(float);

    float* h_A;
    h_A = new float[nElem];

    initialData(h_A, nElem);

    float* d_A;
    cudaMalloc((float**)&d_A, nBytes);

    cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);

    // register call back
    cudaStreamAddCallback(0, data_cp_callback, &nBytes, 0);

    delete[] h_A;
    cudaFree (d_A);

    return 0;
}