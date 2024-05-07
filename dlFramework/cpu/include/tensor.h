#pragma once
#include <vector>
#include <cstddef>  // for size_t

class Tensor
{
public:
    // 构造函数， 接受形状向量作为参数
    Tensor(const std::vector<size_t>& shape);

    // 获取 tensor 形状
    const std::vector<size_t>& shape() const;

    // 获取 tensor 中指定位置元素
    float& at(const std::vector<size_t>& indices);
    const float& at(const std::vector<size_t>& indices) const;

    // 获取 tensor 中元素的数量
    size_t size() const;

    float* pointer();   // 数组首指针

    // 拷贝构造函数
    Tensor(const Tensor& other);

    // 移动构造函数
    Tensor(Tensor&& other) noexcept;    // noexcept 告诉编译器该函数不抛出异常

    // 拷贝赋值运算符
    Tensor& operator=(const Tensor& other);

    // 移动赋值运算符
    Tensor& operator=(Tensor&& other) noexcept; // 使用方法是 tensorA = std::move(tensorB)

    void zeroInit();
    void incrementalInit();

private:
    std::vector<size_t> shape_;
    std::vector<float> data_;

    // 将多维索引转换为平坦化的一维索引
    size_t flatten_index(const std::vector<size_t>& indices) const;
};