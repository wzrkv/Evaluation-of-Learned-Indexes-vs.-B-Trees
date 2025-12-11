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

    // keys must be sorted (SOSD data is already sorted)
    void train(const std::vector<std::uint64_t>& keys);

    // Lookup key in keys; on success return true and write position to pos
    bool search(const std::vector<std::uint64_t>& keys,
                std::uint64_t key,
                std::size_t& pos) const;

    // Rough memory usage estimate
    std::size_t memory_usage_bytes() const;

private:
    std::size_t num_leaves_;
    LinearModel root_;
    std::vector<LinearModel> leaves_;

    // Ordinary least squares fit: y ≈ a * x + b
    static void fit_linear(const std::vector<std::uint64_t>& x,
                           const std::vector<std::size_t>& y,
                           double& a, double& b);
};
