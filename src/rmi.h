#pragma once
#include <vector>
#include <cstddef>
#include <cstdint>

struct LinearModel {
    double a;
    double b;
    std::size_t start_idx;
    std::size_t end_idx;   // exclusive
    std::size_t max_error;
};

class RMI {
public:
    explicit RMI(std::size_t num_leaves = 64);

    // keys 必须是排好序的（SOSD 数据已经是 sorted）
    void train(const std::vector<std::uint64_t>& keys);

    // 在 keys 上查找 key，找到返回 true 并把位置写到 pos
    bool search(const std::vector<std::uint64_t>& keys,
                std::uint64_t key,
                std::size_t& pos) const;

    // 粗略估计内存占用
    std::size_t memory_usage_bytes() const;

private:
    std::size_t num_leaves_;
    LinearModel root_;
    std::vector<LinearModel> leaves_;

    // 简单线性回归：拟合 y ≈ a * x + b
    static void fit_linear(const std::vector<std::uint64_t>& x,
                           const std::vector<std::size_t>& y,
                           double& a, double& b);
};
