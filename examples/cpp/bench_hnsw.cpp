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
    int ef_search = 500;
    float threshold = 8.0;

    nlohmann::json hnsw_parameters{{"max_degree", max_degree},
                                   {"ef_construction", ef_construction},
                                   {"ef_search", ef_search},
                                   {"use_static", true}};
    nlohmann::json index_parameters{
        {"dtype", "float32"}, {"metric_type", "l2"}, {"dim", dim}, {"hnsw", hnsw_parameters}};

    std::shared_ptr<vsag::Index> hnsw;
    if (auto index = vsag::Factory::CreateIndex("hnsw", index_parameters.dump());
        index.has_value()) {
        hnsw = index.value();
    } else {
        std::cout << "Build HNSW Error" << std::endl;
        return 0;
    }

    if (const auto num = hnsw->Build(dataset); num.has_value()) {
        std::cout << "After Build(), Index constains: " << hnsw->GetNumElements() << std::endl;
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
        for (int i = 0; i < query_num; i++) {
            auto query = vsag::Dataset::Make();
            query->NumElements(1)->Dim(dim)->Float32Vectors(queries.data() + i * dim)->Owner(false);

            nlohmann::json parameters{
                {"hnsw", {{"ef_search", ef_search}}},
            };
            int64_t k = 10;
            if (auto result = hnsw->KnnSearch(query, k, parameters.dump()); result.has_value()) {
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
                  << "Memory Uasage:" << hnsw->GetMemoryUsage() / 1024.0 << " KB" << std::endl;
        std::cout << "Recall: " << recall << std::endl;
        std::cout << hnsw->GetStats() << std::endl;
    }

    return 0;
}