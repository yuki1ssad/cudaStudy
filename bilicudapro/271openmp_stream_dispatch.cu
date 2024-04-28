#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <cstring>
#include <omp.h>

#include "00common.h"

#define NSTREAM 4

__device__ void kernel_func()
{
    double sum = 0.0;
    long i = 999999;
    while (i > 0)
    {
        for (long j = 0; j < 999999; ++j)
        {
            sum = sum + tan(0.1) * tan(0.1);
        }
        i -= 1;
    }
    
}

__global__ void kernel_1(int stream)
{
    if (0 == threadIdx.x)
    {
        printf("kernel_1 is executed in stream_%d\n", stream);
    }
    kernel_func();
}

__global__ void kernel_2(int stream)
{
    if (0 == threadIdx.x)
    {
        printf("kernel_2 is executed in stream_%d\n", stream);
    }
    kernel_func();
}

__global__ void kernel_3(int stream)
{
    if (0 == threadIdx.x)
    {
        printf("kernel_3 is executed in stream_%d\n", stream);
    }
    kernel_func();
}

__global__ void kernel_4(int stream)
{
    if (0 == threadIdx.x)
    {
        printf("kernel_4 is executed in stream_%d\n", stream);
    }
    kernel_func();
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

    float elapsed_time;
    // allocate and initialize an array of stream handles
    int n_streams = NSTREAM;
    cudaStream_t *streams = new cudaStream_t[n_streams];
    for (int i = 0; i < n_streams; ++i)
    {
        ErrorCheck(cudaStreamCreate(&(streams[i])), __FILE__, __LINE__);
    }
    
    // set execution configuration
    dim3 block (1);
    dim3 grid (1);

    // creat events
    cudaEvent_t start, stop;
    ErrorCheck(cudaEventCreate(&start), __FILE__, __LINE__);
    ErrorCheck(cudaEventCreate(&stop), __FILE__, __LINE__);

    // record start event
    ErrorCheck(cudaEventRecord(start, 0), __FILE__, __LINE__);

    // execute kernels
    // dispatch with OpenMP
    omp_set_num_threads(NSTREAM);
    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        kernel_1<<<grid, block, 0, streams[thread_id]>>>(thread_id);
        kernel_2<<<grid, block, 0, streams[thread_id]>>>(thread_id);
        kernel_3<<<grid, block, 0, streams[thread_id]>>>(thread_id);
        kernel_4<<<grid, block, 0, streams[thread_id]>>>(thread_id);
    }

    // record stop event
    ErrorCheck(cudaEventRecord(stop, 0), __FILE__, __LINE__);

    std::cout << "Begin to synchronize" << std::endl;
    ErrorCheck(cudaEventSynchronize(stop), __FILE__, __LINE__);

    // calculate elapsed time
    ErrorCheck(cudaEventElapsedTime(&elapsed_time, start, stop), __FILE__, __LINE__);
    std::cout << "Measured time for parallel execution = " << elapsed_time << " ms" << std::endl;

    // release all stream
    for(int i = 0; i < n_streams; ++i)
    {
        ErrorCheck(cudaStreamDestroy(streams[i]), __FILE__, __LINE__);
    }

    delete[] streams;

    ErrorCheck(cudaEventDestroy(start), __FILE__, __LINE__);
    ErrorCheck(cudaEventDestroy(stop), __FILE__, __LINE__);
    ErrorCheck(cudaDeviceReset(), __FILE__, __LINE__);

    return 0;
}