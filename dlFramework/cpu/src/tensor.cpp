#include "tensor.h"
#include <stdexcept>    // for std::out_of_range
#include <numeric>      // for std::accumulate

Tensor::Tensor(const std::vector<size_t>& shape) : shape_(shape)
{
    size_t total_sz = 1;
    for (auto dim : shape_)
    {
        total_sz *= dim;
    }
    data_.resize(total_sz); // 容器的大小调整为 total_sz
}

const std::vector<size_t>& Tensor::shape() const
{
    return shape_;
}

size_t Tensor::size() const
{
    return data_.size();
}

float* Tensor::pointer()
{
    return data_.data();
}

// 将多维索引转换为平坦化的一维索引
size_t Tensor::flatten_index(const std::vector<size_t>& indices) const
{
    size_t flat_index = 0;
    size_t multiplier = 1;
    for (int i = shape_.size() - 1; i > 0; --i)
    {
        flat_index += indices[i] * multiplier;
        multiplier *= shape_[i];
    }
    return flat_index;
}

// 获取 tensor 中指定位置元素
float& Tensor::at(const std::vector<size_t>& indices)
{
    // 检查越界
    if (indices.size() != shape_.size())
    {
        throw std::out_of_range("Index dimensions do not match tensor shape.");
    }
    for (size_t i = 0; i < shape_.size(); ++i)
    {
        if (indices[i] >= shape_[i])
        {
            throw std::out_of_range("Index out of range.");
        }
    }
    return data_[flatten_index(indices)];
}


const float& Tensor::at(const std::vector<size_t>& indices) const
{
    return const_cast<Tensor*>(this)->at(indices);
}

// 拷贝构造函数
Tensor::Tensor(const Tensor& other)
: shape_(other.shape_), data_(other.data_)
{}

// 移动构造函数
Tensor::Tensor(Tensor&& other) noexcept
: shape_(std::move(other.shape_)), data_(std::move(other.data_))
{}

// 拷贝赋值运算符
Tensor& Tensor::operator=(const Tensor& other)
{
    if (this != &other)
    {
        shape_ = other.shape_;
        data_ = other.data_;
    }
    return *this;
}

// 移动赋值运算符
Tensor& Tensor::operator=(Tensor&& other) noexcept
{
    if (this != &other)
    {
        shape_ = std::move(other.shape_);
        data_ = std::move(other.data_);
    }
    return *this;
}

void Tensor::zeroInit()
{
    std::fill(data_.begin(), data_.end(), 0.0f);
}

void Tensor::incrementalInit()
{
    for (int i = 0; i < size(); ++i)
    {
        data_[i] = static_cast<float>(i);
    }
}
