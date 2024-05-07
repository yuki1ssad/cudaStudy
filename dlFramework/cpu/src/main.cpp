#include "linear.h"
#include "tensor.h"
#include "omp_matrix.h"
#include <omp.h>
#include <iostream>

int main()
{
    size_t batchsize = 4;
    size_t M = 64;
    size_t N = 3;

    Tensor tensorA({batchsize, M});
    tensorA.incrementalInit();

    Tensor tensorC({batchsize, N});

    double st, ela;
    st = omp_get_wtime();
    std::vector<size_t> linear_size({M, N});
    Linear mlp(linear_size);

    tensorC = mlp.linearTransform(tensorA);
    
    ela = omp_get_wtime() - st;
    std::cout << "time: " << ela << std::endl;


    std::cout << "tensorC elements: " << std::endl;
    for (size_t i = 0; i < tensorC.shape()[0]; ++i)
    {
        for (size_t j = 0; j < tensorC.shape()[1]; ++j)
        {
            std::cout << tensorC.at({i, j}) << " ";
        }
        std::cout << std::endl;
    }
    return 0;
}