#include "tensor.h"
#include "omp_matrix.h"
#include <omp.h>
#include <iostream>

int main()
{
    Tensor tensorA({2, 3});
    tensorA.at({0, 0}) = 1.0;
    tensorA.at({0, 1}) = 2.0;
    tensorA.at({0, 2}) = 3.0;
    tensorA.at({1, 0}) = 4.0;
    tensorA.at({1, 1}) = 5.0;
    tensorA.at({1, 2}) = 6.0;

    Tensor tensorB({3, 2});
    tensorB.at({0, 0}) = 1.0;
    tensorB.at({0, 1}) = 2.0;
    tensorB.at({1, 0}) = 3.0;
    tensorB.at({1, 1}) = 4.0;
    tensorB.at({2, 0}) = 5.0;
    tensorB.at({2, 1}) = 6.0;

    Tensor tensorC({2, 2});

    double st, ela;
    st = omp_get_wtime();
    #pragma omp parallel
    {
        omp_matrix(tensorA.pointer(), tensorB.pointer(), tensorC.pointer(), 2, 3, 2);
    }
    ela = omp_get_wtime() - st;
    std::cout << "time: " << ela << std::endl;

    std::cout << "tensorA shape: ";
    for (size_t dim : tensorA.shape())
    {
        std::cout << dim << " ";
    }
    std::cout << std::endl;
    std::cout << "tensorA size: " << tensorA.size() << std::endl;

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