#pragma once
#include <vector>
#include <cstddef>
#include <cstdint>

struct BPTreeNode {
    bool is_leaf;
    std::size_t order;

    std::uint64_t min_key;

    std::vector<std::uint64_t> keys;
    std::vector<std::size_t> children;
    std::vector<BPTreeNode*> child_ptrs;

    BPTreeNode* next;

    explicit BPTreeNode(bool leaf, std::size_t ord)
        : is_leaf(leaf), order(ord), min_key(0), next(nullptr) {}
};

class BPTree {
public:
    explicit BPTree(std::size_t order = 64);
    ~BPTree();

    void bulk_load(const std::vector<std::uint64_t>& keys);
    bool search(std::uint64_t key, std::size_t& pos) const;
    std::size_t memory_usage_bytes() const;

private:
    std::size_t order_;
    BPTreeNode* root_;

    void free_node(BPTreeNode* node);
    std::size_t count_nodes(BPTreeNode* node) const;
};
