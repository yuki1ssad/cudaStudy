#pragma once
#include <vector>
#include "tensor.h"

class Linear
{
private:
    /* data */
public:
    Linear(std::vector<size_t> shape);  // 构造函数，接受形式如(M,N)的shape作为参数

    // 正向传播函数， 接收一个类型为 Tensor 的类作为输入
    Tensor linearTransform(Tensor &Input);

    std::vector<size_t> shape_;
    std::vector<float> weight_; // 权重的具体元素，形状为shape=[M,N]
    std::vector<float> bias_;   // 形状为[N]    
};