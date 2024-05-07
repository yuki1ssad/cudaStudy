#include "linear.h"
#include "tensor.h"
#include "omp_matrix.h"
#include <cassert>  // for assert

Linear::Linear(std::vector<size_t> shape) : shape_(shape)
{
    weight_.resize(shape_[0] * shape_[1], 1.0f);    // 初始化权重数据全是1
    bias_.resize(shape_[1], 0.3333f);                // 初始化偏置元素全是0.3333
}

// 正向传播函数， 接收一个类型为 Tensor 的类作为输入
Tensor Linear::linearTransform(Tensor &Input)
{
    // Input的形状为[batchsize, M], weight形状为[M,N]，bias形状为[N]
    std::vector<size_t> output_shape({Input.shape()[0], shape_[1]});
    Tensor Output(output_shape);

    // #pragma omp parallel
    // {
    //     omp_matrix(Input.pointer(), weight_.data(), Output.pointer(), Input.shape()[0], shape_[0], shape_[1]);
    // }

    omp_matrix(Input.pointer(), weight_.data(), Output.pointer(), Input.shape()[0], shape_[0], shape_[1]);

    // Add the bias to the output tensor.
    for (size_t b = 0; b < Input.shape()[0]; b++)
    { // 定义偏置计算过程
        for (size_t i = 0; i < bias_.size(); ++i)
        {
            Output.pointer()[b * shape_[1] + i] += bias_[i];
        }
    }

    return Output;
}