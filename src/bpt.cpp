#include "bpt.h"

#include <algorithm>

BPTree::BPTree(std::size_t order) : order_(order), root_(nullptr) {}

BPTree::~BPTree() {
    free_node(root_);
}

void BPTree::free_node(BPTreeNode* node) {
    if (!node) return;
    if (!node->is_leaf) {
        for (BPTreeNode* ch : node->child_ptrs) {
            free_node(ch);
        }
    }
    delete node;
}

std::size_t BPTree::count_nodes(BPTreeNode* node) const {
    if (!node) return 0;
    std::size_t cnt = 1;
    if (!node->is_leaf) {
        for (BPTreeNode* ch : node->child_ptrs) {
            cnt += count_nodes(ch);
        }
    }
    return cnt;
}

void BPTree::bulk_load(const std::vector<std::uint64_t>& keys) {
    // 清空旧树
    free_node(root_);
    root_ = nullptr;

    std::size_t n = keys.size();
    if (n == 0) return;

    // 构建叶子层
    std::vector<BPTreeNode*> leaves;
    std::size_t i = 0;
    while (i < n) {
        BPTreeNode* leaf = new BPTreeNode(true, order_);
        std::size_t end = std::min(i + order_, n);
        leaf->keys.reserve(end - i);
        leaf->children.reserve(end - i);
        for (std::size_t j = i; j < end; ++j) {
            leaf->keys.push_back(keys[j]);
            leaf->children.push_back(j); // 保存原始位置
        }
        leaf->min_key = leaf->keys.front();  // 叶子子树最小key
        leaves.push_back(leaf);
        i = end;
    }

    for (std::size_t j = 0; j + 1 < leaves.size(); ++j) {
        leaves[j]->next = leaves[j + 1];
    }

    // 自底向上构建内部节点
    std::vector<BPTreeNode*> level = leaves;
    while (level.size() > 1) {
        std::vector<BPTreeNode*> new_level;
        std::size_t idx = 0;
        while (idx < level.size()) {
            BPTreeNode* parent = new BPTreeNode(false, order_);
            std::size_t group_end = std::min(idx + order_, level.size());
            parent->child_ptrs.reserve(group_end - idx);

            for (std::size_t k = idx; k < group_end; ++k) {
                parent->child_ptrs.push_back(level[k]);
            }
            parent->min_key = parent->child_ptrs.front()->min_key;

            // 分割key
            for (std::size_t child_idx = 1; child_idx < parent->child_ptrs.size(); ++child_idx) {
                parent->keys.push_back(parent->child_ptrs[child_idx]->min_key);
            }

            new_level.push_back(parent);
            idx = group_end;
        }
        level = std::move(new_level);
    }

    root_ = level[0];
}

bool BPTree::search(std::uint64_t key, std::size_t& pos) const {
    BPTreeNode* node = root_;
    if (!node) return false;

    // 从root向下
    while (!node->is_leaf) {
        const auto& keys = node->keys;
        std::size_t lo = 0, hi = keys.size();
        while (lo < hi) {
            std::size_t mid = (lo + hi) / 2;
            if (key < keys[mid]) hi = mid;
            else lo = mid + 1;
        }
        std::size_t child_idx = lo;
        if (child_idx >= node->child_ptrs.size())
            child_idx = node->child_ptrs.size() - 1;
        node = node->child_ptrs[child_idx];
    }

    // binary search
    const auto& keys_leaf = node->keys;
    const auto& pos_leaf  = node->children;
    std::size_t lo = 0, hi = keys_leaf.size();
    while (lo < hi) {
        std::size_t mid = (lo + hi) / 2;
        if (key < keys_leaf[mid]) {
            hi = mid;
        } else if (key > keys_leaf[mid]) {
            lo = mid + 1;
        } else {
            pos = pos_leaf[mid];
            return true;
        }
    }
    return false;
}

std::size_t BPTree::memory_usage_bytes() const {
    std::size_t nodes = count_nodes(root_);
    return nodes * 512;
}
