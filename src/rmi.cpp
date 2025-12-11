#include "rmi.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

RMI::RMI(std::size_t num_leaves)
    : num_leaves_(num_leaves), root_{0.0, 0.0, 0, 0, 0} {}

void RMI::fit_linear(const std::vector<std::uint64_t>& x,
                     const std::vector<std::size_t>& y,
                     double& a, double& b) {
    std::size_t n = x.size();
    if (n == 0) {
        a = 0.0;
        b = 0.0;
        return;
    }
    long double sum_x = 0.0L, sum_y = 0.0L;
    long double sum_x2 = 0.0L, sum_xy = 0.0L;
    for (std::size_t i = 0; i < n; ++i) {
        long double xi = static_cast<long double>(x[i]);
        long double yi = static_cast<long double>(y[i]);
        sum_x += xi;
        sum_y += yi;
        sum_x2 += xi * xi;
        sum_xy += xi * yi;
    }
    long double n_ld = static_cast<long double>(n);
    long double denom = n_ld * sum_x2 - sum_x * sum_x;
    if (std::fabs(static_cast<double>(denom)) < 1e-12) {
        a = 0.0;
        b = static_cast<double>(sum_y / n_ld);
    } else {
        long double num = n_ld * sum_xy - sum_x * sum_y;
        long double a_ld = num / denom;
        long double b_ld = (sum_y - a_ld * sum_x) / n_ld;
        a = static_cast<double>(a_ld);
        b = static_cast<double>(b_ld);
    }
}

void RMI::train(const std::vector<std::uint64_t>& keys) {
    std::size_t n = keys.size();
    if (n == 0) {
        throw std::runtime_error("RMI::train: empty keys");
    }

    // 1) root 模型：所有 keys -> index
    std::vector<std::uint64_t> x_root(keys);
    std::vector<std::size_t> y_root(n);
    for (std::size_t i = 0; i < n; ++i) y_root[i] = i;

    double a_root = 0.0, b_root = 0.0;
    fit_linear(x_root, y_root, a_root, b_root);
    root_.a = a_root;
    root_.b = b_root;
    root_.start_idx = 0;
    root_.end_idx = n;
    root_.max_error = 0;

    // 2) 用 root 的预测给每个 key 分配一个 leaf bucket
    leaves_.clear();
    leaves_.resize(num_leaves_, {0.0, 0.0, 0, 0, 0});
    std::vector<std::vector<std::size_t>> buckets(num_leaves_);

    auto clamp_pos = [n](long double p) -> std::size_t {
        if (p < 0) return 0;
        if (p >= static_cast<long double>(n)) return n - 1;
        return static_cast<std::size_t>(p);
    };

    for (std::size_t i = 0; i < n; ++i) {
        long double pred = static_cast<long double>(a_root) *
                           static_cast<long double>(keys[i]) +
                           static_cast<long double>(b_root);
        std::size_t pos = clamp_pos(pred);
        std::size_t leaf_id = (pos * num_leaves_) / n;
        if (leaf_id >= num_leaves_) leaf_id = num_leaves_ - 1;
        buckets[leaf_id].push_back(i);
    }

    // 3) 每个 leaf 拟合自己的线性模型，并计算 max_error / start / end
    for (std::size_t leaf_id = 0; leaf_id < num_leaves_; ++leaf_id) {
        auto& idxs = buckets[leaf_id];
        if (idxs.empty()) {
            leaves_[leaf_id] = {0.0, 0.0, 0, 0, 0};
            continue;
        }
        std::vector<std::uint64_t> x;
        std::vector<std::size_t> y;
        x.reserve(idxs.size());
        y.reserve(idxs.size());
        for (std::size_t idx : idxs) {
            x.push_back(keys[idx]);
            y.push_back(idx);
        }
        double a = 0.0, b = 0.0;
        fit_linear(x, y, a, b);

        std::size_t max_err = 0;
        std::size_t start_idx = y[0];
        std::size_t end_idx = y[0];
        for (std::size_t i = 0; i < x.size(); ++i) {
            long double pred = static_cast<long double>(a) *
                               static_cast<long double>(x[i]) +
                               static_cast<long double>(b);
            std::size_t pos = clamp_pos(pred);
            std::size_t true_pos = y[i];
            std::size_t err = (pos > true_pos) ? (pos - true_pos) : (true_pos - pos);
            if (err > max_err) max_err = err;
            if (true_pos < start_idx) start_idx = true_pos;
            if (true_pos > end_idx) end_idx = true_pos;
        }
        leaves_[leaf_id].a = a;
        leaves_[leaf_id].b = b;
        leaves_[leaf_id].start_idx = start_idx;
        leaves_[leaf_id].end_idx = end_idx + 1; // end 是开区间
        leaves_[leaf_id].max_error = max_err;
    }
}

bool RMI::search(const std::vector<std::uint64_t>& keys,
                 std::uint64_t key,
                 std::size_t& pos) const {
    std::size_t n = keys.size();
    if (n == 0) return false;

    auto clamp_pos = [n](long double p) -> std::size_t {
        if (p < 0) return 0;
        if (p >= static_cast<long double>(n)) return n - 1;
        return static_cast<std::size_t>(p);
    };

    // 1) root 预测 -> leaf id
    long double pred_root = static_cast<long double>(root_.a) *
                            static_cast<long double>(key) +
                            static_cast<long double>(root_.b);
    std::size_t pos_root = clamp_pos(pred_root);
    std::size_t leaf_id = (pos_root * num_leaves_) / n;
    if (leaf_id >= num_leaves_) leaf_id = num_leaves_ - 1;

    const auto& leaf = leaves_[leaf_id];

    // 2) leaf 内预测
    long double pred_leaf = static_cast<long double>(leaf.a) *
                            static_cast<long double>(key) +
                            static_cast<long double>(leaf.b);
    std::size_t p = clamp_pos(pred_leaf);

    std::size_t lo = leaf.start_idx;
    std::size_t hi = (leaf.end_idx == 0) ? 0 : leaf.end_idx - 1;
    if (leaf.max_error > 0) {
        std::size_t left = (p > leaf.max_error) ? (p - leaf.max_error) : 0;
        std::size_t right = std::min(p + leaf.max_error, n - 1);
        if (left > lo) lo = left;
        if (right < hi) hi = right;
    }

    // 3) 在 [lo, hi] 范围内二分查找
    while (lo <= hi) {
        std::size_t mid = (lo + hi) / 2;
        std::uint64_t mid_key = keys[mid];
        if (key < mid_key) {
            if (mid == 0) break;
            hi = mid - 1;
        } else if (key > mid_key) {
            lo = mid + 1;
        } else {
            pos = mid;
            return true;
        }
    }
    return false;
}

std::size_t RMI::memory_usage_bytes() const {
    return sizeof(LinearModel) * (1 + leaves_.size());
}
