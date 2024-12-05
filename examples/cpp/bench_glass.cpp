#include <cstdint>
#include <exception>
#include <fstream>
#include <iostream>
#include <memory>
#include <string_view>
#include <unordered_set>

#include "nlohmann/json.hpp"
#include "vsag/dataset.h"
#include "vsag/vsag.h"

class HighPrecisionTimer {
public:
    HighPrecisionTimer() : total_duration(0), running(false) {
    }

    // 开始计时
    void
    start() {
        if (!running) {
            start_time = std::chrono::high_resolution_clock::now();
            running = true;
        }
    }

    // 停止计时并累加时间
    void
    stop() {
        if (running) {
            auto end_time = std::chrono::high_resolution_clock::now();
            total_duration +=
                std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count();
            running = false;
        }
    }

    // 获取累积时间（以纳秒为单位）
    long long
    getTotalTimeNanoseconds() const {
        return total_duration;
    }

    // 获取累积时间（以微秒为单位）
    double
    getTotalTimeMicroseconds() const {
        return total_duration / 1000.0;
    }

    // 获取累积时间（以毫秒为单位）
    double
    getTotalTimeMilliseconds() const {
        return total_duration / 1000000.0;
    }

    // 获取累积时间（以秒为单位）
    double
    getTotalTimeSeconds() const {
        return total_duration / 1000000000.0;
    }

    // 重置计时器
    void
    reset() {
        total_duration = 0;
        running = false;
    }

private:
    std::chrono::high_resolution_clock::time_point start_time;
    long long total_duration;  // 总耗时，单位：纳秒
    bool running;              // 记录计时器是否正在运行
};

std::vector<float>
load_fvecs(std::string_view file_path, uint32_t& vec_dim, uint32_t& max_elements) {
    std::ifstream file(std::string(file_path), std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << file_path;
        std::terminate();
    }
    std::vector<float> data;
    max_elements = 0;
    while (!file.eof()) {
        int dim;
        file.read(reinterpret_cast<char*>(&dim), sizeof(int));
        if (file.eof())
            break;
        std::vector<float> vec(dim);
        vec_dim = dim;
        file.read(reinterpret_cast<char*>(vec.data()), dim * sizeof(float));
        data.insert(data.end(), vec.begin(), vec.end());
        ++max_elements;
    }
    std::cout << "finish loading " << file_path << "\nmax_elements: " << max_elements << std::endl;

    return data;
}

std::vector<std::vector<int>>
load_ivecs(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + filename);
    }
    std::vector<std::vector<int>> data;
    while (!file.eof()) {
        int dim;
        file.read(reinterpret_cast<char*>(&dim), sizeof(int));
        if (file.eof())
            break;
        std::vector<int> vec(dim);
        file.read(reinterpret_cast<char*>(vec.data()), dim * sizeof(int));
        data.push_back(std::move(vec));
    }
    std::cout << "finish loading " << filename << "\nmax_element: " << data.size() << std::endl;
    return data;
}

float
compute_recall(const std::vector<std::vector<int>>& groundtruth,
               const int64_t* predictions,
               int query_id,
               int k) {
    float recall = 0.0f;

    int hits = 0;
    std::unordered_set<int> gt_set(groundtruth[query_id].begin(),
                                   groundtruth[query_id].begin() + k);
    for (int j = 0; j < k; ++j) {
        if (gt_set.find(predictions[j]) != gt_set.end()) {
            ++hits;
        }
    }
    recall += static_cast<float>(hits) / k;
    return recall;
}
int
main() {
    uint32_t max_elements, dim;

    auto data = load_fvecs("/home/yinshi/dataset/siftsmall/siftsmall_base.fvecs", dim, max_elements);
    uint32_t vec_size = dim * sizeof(float);

    int64_t* ids = new int64_t[max_elements];

    for (int32_t i = 0; i < max_elements; ++i) {
        ids[i] = i;
    }

    auto dataset = vsag::Dataset::Make();
    dataset->Dim(dim)
        ->NumElements(max_elements - 1)
        ->Ids(ids)
        ->Float32Vectors(data.data())
        ->Owner(false);

    int max_degree = 24;  // Tightly connected with internal dimensionality of the data
    // strongly affects the memory consumption
    int ef_construction = 500;  // Controls index search speed/build speed tradeoff
    int ef_search = 200;
    float threshold = 8.0;

    nlohmann::json glass_parameters{{"max_degree", max_degree},
                                    {"ef_construction", ef_construction},
                                    {"ef_search", ef_search},
                                    {"use_static", false}};
    nlohmann::json index_parameters{
        {"dtype", "float32"}, {"metric_type", "l2"}, {"dim", dim}, {"glass", glass_parameters}};

    std::shared_ptr<vsag::Index> glass;
    if (auto index = vsag::Factory::CreateIndex("glass", index_parameters.dump());
        index.has_value()) {
        glass = index.value();
    } else {
        std::cout << "Build HNSW Error" << std::endl;
        return 0;
    }

    if (const auto num = glass->Build(dataset); num.has_value()) {
        std::cout << "After Build(), Index constains: " << glass->GetNumElements() << std::endl;
    } else if (num.error().type == vsag::ErrorType::INTERNAL_ERROR) {
        std::cerr << "Failed to build index: internalError" << std::endl;
        exit(-1);
    }
    uint32_t query_num;
    auto queries = load_fvecs("/home/yinshi/dataset/siftsmall/siftsmall_query.fvecs", dim, query_num);

    auto ground_truth = load_ivecs("/home/yinshi/dataset/siftsmall/siftsmall_groundtruth.ivecs");

    float correct = 0;
    float recall = 0;
    {
        HighPrecisionTimer timer;
        for (int i = 0; i < query_num; i++) {
            auto query = vsag::Dataset::Make();
            query->NumElements(1)->Dim(dim)->Float32Vectors(queries.data() + i * dim)->Owner(false);

            nlohmann::json parameters{
                {"glass", {{"ef_search", ef_search}}},
            };
            int64_t k = 10;
            timer.start();
            if (auto result = glass->KnnSearch(query, k, parameters.dump()); result.has_value()) {
                timer.stop();
                correct += compute_recall(ground_truth, result.value()->GetIds(), i, k);
            } else if (result.error().type == vsag::ErrorType::INTERNAL_ERROR) {
                std::cerr << "failed to perform knn search on index" << std::endl;
            }
            if (not(i % 10000)) {
                std::cout << i << std::endl;
            }
        }
        recall = correct / query_num;
        std::cout << std::fixed << std::setprecision(5)
                  << "Memory Uasage:" << glass->GetMemoryUsage() / 1024.0 << " KB" << std::endl;
        std::cout << "Recall: " << recall << std::endl;
        std::cout << "time cost: " << timer.getTotalTimeMilliseconds() << " ms" << std::endl;
        std::cout << "time avg cost: " << timer.getTotalTimeMilliseconds() / query_num << " ms"
                  << std::endl;
    }

    return 0;
}