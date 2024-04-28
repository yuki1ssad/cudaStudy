#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <cstring>
#include <omp.h>

#include "00common.h"

int main(int argc, char** argv)
{
    omp_set_num_threads(3);
    #pragma omp parallel
    {
        std::cout << "thread is running\n" <<std::endl;
    }
    return 0;
}

/*
    nvcc -Xcompiler -fopenmp 261openMp_usage.cu 
*/