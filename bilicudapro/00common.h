#include <sys/time.h>
#include <cuda_runtime.h>
#include <iostream>
#include <chrono>

cudaError_t ErrorCheck(cudaError_t status, const char* filename, int lineNumber)
{
    if (status != cudaSuccess)
    {
        std::cout << "CUDA API error:\r\ncode=" << status << ", name=" << cudaGetErrorName(status) << ", description=" << cudaGetErrorString(status) << ", line=" << lineNumber << std::endl;
    }
    return status;
}

inline double GetCPUSecond()
{
    struct timeval tp;
    struct timezone tzp;
    int i = gettimeofday(&tp, &tzp);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}