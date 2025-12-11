#pragma once
#include <vector>
#include <cstddef>
#include <cstdint>

struct BPTreeNode {
    bool is_leaf;
    std::size_t order;

    // 新增：整棵子树的最小 key，用来在高层做正确的分割
    std::uint64_t min_key;

    // 对于叶子：keys = 实际数据 key，children = 原始数组下标
    // 对于内部节点：keys = 分割 key（右孩子子树的 min_key），child_ptrs = 子节点指针
    std::vector<std::uint64_t> keys;
    std::vector<std::size_t> children;
    std::vector<BPTreeNode*> child_ptrs;

    BPTreeNode* next;  // 叶子链表（可选）

    explicit BPTreeNode(bool leaf, std::size_t ord)
        : is_leaf(leaf), order(ord), min_key(0), next(nullptr) {}
};

class BPTree {
public:
    explicit BPTree(std::size_t order = 64);
    ~BPTree();

    // 从排好序的 keys 构建静态 B+ 树
    void bulk_load(const std::vector<std::uint64_t>& keys);

    // 查找 key，对应原始数组位置写到 pos
    bool search(std::uint64_t key, std::size_t& pos) const;

    // 粗略估计内存
    std::size_t memory_usage_bytes() const;

private:
    std::size_t order_;
    BPTreeNode* root_;

    void free_node(BPTreeNode* node);
    std::size_t count_nodes(BPTreeNode* node) const;
};
