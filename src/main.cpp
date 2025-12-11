#include <algorithm>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "rmi.h"
#include "bpt.h"

using std::cout;
using std::cerr;
using std::endl;

// ------------- Data loading & query generation -------------

// Load uint64_t keys from a binary file
std::vector<std::uint64_t> load_dataset(const std::string& path, std::size_t max_keys = 0) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Cannot open file: " + path);
    }
    in.seekg(0, std::ios::end);
    std::size_t bytes = static_cast<std::size_t>(in.tellg());
    in.seekg(0, std::ios::beg);
    std::size_t total = bytes / sizeof(std::uint64_t);
    if (max_keys > 0 && max_keys < total) {
        total = max_keys;
    }
    std::vector<std::uint64_t> keys(total);
    in.read(reinterpret_cast<char*>(keys.data()), total * sizeof(std::uint64_t));
    if (!in) {
        throw std::runtime_error("Failed to read file: " + path);
    }
    return keys;
}

// Sample queries uniformly from existing keys
std::vector<std::uint64_t> generate_queries(const std::vector<std::uint64_t>& keys,
                                            std::size_t num_queries) {
    std::mt19937_64 rng(42);
    std::uniform_int_distribution<std::size_t> dist(0, keys.size() - 1);
    std::vector<std::uint64_t> qs;
    qs.reserve(num_queries);
    for (std::size_t i = 0; i < num_queries; ++i) {
        qs.push_back(keys[dist(rng)]);
    }
    return qs;
}

// ------------- Stats & benchmarking -------------

struct Stats {
    double mean_ns;
    double p95_ns;
    double p99_ns;
};

Stats compute_stats(std::vector<long long>& latencies_ns) {
    std::size_t n = latencies_ns.size();
    if (n == 0) return {0, 0, 0};
    std::vector<long long> v = latencies_ns;
    std::sort(v.begin(), v.end());
    double mean = std::accumulate(v.begin(), v.end(), 0.0) / static_cast<double>(n);
    std::size_t idx95 = static_cast<std::size_t>(0.95 * n);
    if (idx95 >= n) idx95 = n - 1;
    std::size_t idx99 = static_cast<std::size_t>(0.99 * n);
    if (idx99 >= n) idx99 = n - 1;
    double p95 = static_cast<double>(v[idx95]);
    double p99 = static_cast<double>(v[idx99]);
    return {mean, p95, p99};
}

Stats benchmark_bpt(const std::vector<std::uint64_t>& keys,
                    BPTree& tree,
                    const std::vector<std::uint64_t>& queries) {
    std::vector<long long> latencies;
    latencies.reserve(queries.size());
    using clock = std::chrono::high_resolution_clock;

    for (auto q : queries) {
        auto t0 = clock::now();
        std::size_t pos = 0;
        bool ok = tree.search(q, pos);
        auto t1 = clock::now();
        auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
        latencies.push_back(ns);
        (void)ok; // sanity is checked separately
    }
    return compute_stats(latencies);
}

Stats benchmark_rmi(const std::vector<std::uint64_t>& keys,
                    RMI& rmi,
                    const std::vector<std::uint64_t>& queries) {
    std::vector<long long> latencies;
    latencies.reserve(queries.size());
    using clock = std::chrono::high_resolution_clock;

    for (auto q : queries) {
        auto t0 = clock::now();
        std::size_t pos = 0;
        bool ok = rmi.search(keys, q, pos);
        auto t1 = clock::now();
        auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
        latencies.push_back(ns);
        (void)ok;
    }
    return compute_stats(latencies);
}

// ------------- Sanity checks -------------

void sanity_check(const std::vector<std::uint64_t>& keys,
                  BPTree& bpt,
                  RMI& rmi) {
    std::mt19937_64 rng(123);
    std::uniform_int_distribution<std::size_t> dist(0, keys.size() - 1);

    // Check 100 random existing keys
    for (int i = 0; i < 100; ++i) {
        std::size_t idx = dist(rng);
        std::uint64_t k = keys[idx];

        std::size_t pos_b = 0, pos_r = 0;
        bool ok_b = bpt.search(k, pos_b);
        bool ok_r = rmi.search(keys, k, pos_r);

        if (!ok_b || !ok_r || keys[pos_b] != k || keys[pos_r] != k) {
            std::cerr << "[SANITY] mismatch on existing key " << k
                      << " bpt_ok=" << ok_b << " rmi_ok=" << ok_r
                      << " pos_b=" << pos_b << " pos_r=" << pos_r << "\n";
            return;
        }
    }

    // Probe 100 random non-keys (just to ensure no crashes)
    for (int i = 0; i < 100; ++i) {
        std::size_t idx = dist(rng);
        std::uint64_t k = keys[idx] + 1;

        std::size_t pos_b = 0, pos_r = 0;
        bool ok_b = bpt.search(k, pos_b);
        bool ok_r = rmi.search(keys, k, pos_r);
        (void)ok_b;
        (void)ok_r;
    }

    std::cout << "[SANITY] basic checks passed.\n";
}

// ------------- main -------------

int main() {
    try {
        // ===== CSV output: lookup vs build/memory =====
        std::ofstream csv_lookup("results_lookup.csv");
        csv_lookup << "dataset,index,num_keys,num_leaves,metric,mean_ns,p95_ns,p99_ns\n";

        std::ofstream csv_build("results_build.csv");
        csv_build << "dataset,index,num_keys,num_leaves,build_time_s,mem_bytes\n";
        // =============================================

        std::string base = "data/"; // relative to project root

        std::map<std::string, std::string> datasets = {
            {"books", base + "books_200M_uint64"},
            {"fb",    base + "fb_200M_uint64"},
            {"osm",   base + "osm_cellids_200M_uint64"},
            {"wiki",  base + "wiki_ts_200M_uint64"}
        };

        // Change this to 1M / 5M / 10M as needed
        std::size_t max_keys    = 100'000'000;   // e.g., 1'000'000 or 5'000'000
        std::size_t num_queries = 100'000;

        using clock = std::chrono::high_resolution_clock;

        for (const auto& [name, path] : datasets) {
            cout << "\n==== Dataset: " << name << " ====" << endl;
            auto keys = load_dataset(path, max_keys);
            cout << "Loaded " << keys.size() << " keys from " << path << endl;

            // ---- Build B+Tree ----
            BPTree bpt(64);
            auto t0 = clock::now();
            bpt.bulk_load(keys);
            auto t1 = clock::now();
            std::chrono::duration<double> dt_b = t1 - t0;
            double build_time_b = dt_b.count();
            std::size_t mem_b   = bpt.memory_usage_bytes();

            cout << "B+Tree build time: " << build_time_b
                 << " s, approx mem " << mem_b / 1024.0 / 1024.0
                 << " MB" << endl;

            // Write B+Tree build/mem stats (num_leaves left empty)
            csv_build << name << ",BPTree," << keys.size() << ","
                      << "" << ","
                      << build_time_b << ","
                      << mem_b << "\n";

            // ---- Generate queries (shared across all indexes) ----
            auto queries = generate_queries(keys, num_queries);

            // ---- B+Tree lookup benchmark ----
            auto stats_b = benchmark_bpt(keys, bpt, queries);
            cout << "B+Tree lookup: mean=" << stats_b.mean_ns
                 << " ns, p95=" << stats_b.p95_ns
                 << " ns, p99=" << stats_b.p99_ns << " ns" << endl;

            csv_lookup << name << ",BPTree," << keys.size() << ","
                       << "" << ","
                       << "lookup," << stats_b.mean_ns << ","
                       << stats_b.p95_ns << ","
                       << stats_b.p99_ns << "\n";

            // ---- RMI: sweep leaves on books/osm, use 64 elsewhere ----
            std::vector<int> leaf_configs;
            if (name == "books" || name == "osm") {
                leaf_configs = {32, 64, 128, 256};
            } else {
                leaf_configs = {64};
            }

            for (int leaves : leaf_configs) {
                cout << "\n--- RMI with " << leaves << " leaves ---\n";
                RMI rmi(leaves);

                auto t2 = clock::now();
                rmi.train(keys);
                auto t3 = clock::now();
                std::chrono::duration<double> dt_r = t3 - t2;
                double train_time_r = dt_r.count();
                std::size_t mem_r   = rmi.memory_usage_bytes();

                cout << "RMI(" << leaves << ") train time: " << train_time_r
                     << " s, approx mem " << mem_r / 1024.0
                     << " KB" << endl;

                // Write RMI build/mem stats
                csv_build << name << ",RMI," << keys.size() << ","
                          << leaves << ","
                          << train_time_r << ","
                          << mem_r << "\n";

                // Sanity check
                sanity_check(keys, bpt, rmi);

                // RMI lookup benchmark
                auto stats_r = benchmark_rmi(keys, rmi, queries);

                cout << "RMI(" << leaves << ") lookup: mean=" << stats_r.mean_ns
                     << " ns, p95=" << stats_r.p95_ns
                     << " ns, p99=" << stats_r.p99_ns << " ns" << endl;

                csv_lookup << name << ",RMI," << keys.size() << ","
                           << leaves << ","
                           << "lookup," << stats_r.mean_ns << ","
                           << stats_r.p95_ns << ","
                           << stats_r.p99_ns << "\n";
            }
        }
    } catch (const std::exception& e) {
        cerr << "Error: " << e.what() << endl;
        return 1;
    }
    return 0;
}