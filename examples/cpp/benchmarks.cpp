#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <nlohmann/json.hpp>
#include <random>
#include <unordered_set>
#include <vector>

#include "H5Cpp.h"
#include "vsag/errors.h"
#include "vsag/vsag.h"

using namespace H5;

float
compute_recall(const std::vector<std::vector<int>>& groundtruth,
               const int64_t* predictions,
               int query_id,
               int k) {
    float recall = 0.0f;

    int hits = 0;
    std::unordered_set<int> gt_set(groundtruth[query_id].begin(),
                                   groundtruth[query_id].begin() + k);
    // 验证 predictions 和 gt_set 是否对齐
    // std::cout << "Groundtruth Set for query " << query_id << ": ";
    // for (int val : gt_set) std::cout << val << " ";
    // std::cout << std::endl;

    // std::cout << "Predictions for query " << query_id << ": ";
    // for (int j = 0; j < k; ++j) std::cout << predictions[j] << " ";
    // std::cout << std::endl;
    for (int j = 0; j < k; ++j) {
        if (gt_set.find(predictions[j]) != gt_set.end()) {
            ++hits;
        }
    }
    recall += static_cast<float>(hits) / k;
    return recall;
}

int
bench(const std::string& index_type,
      int max_elements,
      int query_elements,
      int dim,
      int ef_search,
      int k,
      nlohmann::json hnsw_parameters,
      std::vector<float>& X_train,
      std::vector<float>& X_test,
      std::vector<std::vector<int>>& ground_truth) {
    std::shared_ptr<int64_t[]> ids(new int64_t[max_elements]);
    for (int32_t i = 0; i < max_elements; ++i) {
        ids[i] = i;
    }

    nlohmann::json index_parameters{
        {"dtype", "float32"}, {"metric_type", "l2"}, {"dim", dim}, {index_type, hnsw_parameters}};
    std::shared_ptr<vsag::Index> index;
    if (auto index = vsag::Factory::CreateIndex(index_type, index_parameters.dump());
        index.has_value()) {
        index = index.value();
    } else {
        std::cout << "Build HNSW Error" << std::endl;
        return 0;
    }
    // Build index
    {
        auto dataset = vsag::Dataset::Make();
        dataset->Dim(dim)
            ->NumElements(max_elements)
            ->Ids(ids.get())
            ->Float32Vectors(X_train.data())
            ->Owner(false);
        if (const auto num = index->Build(dataset); num.has_value()) {
            std::cout << "After Build(), Index constains: " << index->GetNumElements() << std::endl;
        } else if (num.error().type == vsag::ErrorType::INTERNAL_ERROR) {
            std::cerr << "Failed to build index: internalError" << std::endl;
            exit(-1);
        }
    }

    // Query the elements for themselves and measure recall 1@1
    float correct = 0;
    float recall = 0;
    double total_time = 0;
    {
        for (int i = 0; i < query_elements; i++) {
            auto query = vsag::Dataset::Make();
            query->NumElements(1)->Dim(dim)->Float32Vectors(X_test.data() + i * dim)->Owner(false);
            // {
            //   "hnsw": {
            //     "ef_search": 200
            //   }
            // }

            nlohmann::json parameters{
                {index_type, {{"ef_search", ef_search}}},
            };
            // int64_t k = 100;
            auto start = std::chrono::system_clock::now();
            auto result = index->KnnSearch(query, k, parameters.dump());
            auto end = std::chrono::system_clock::now();
            std::chrono::duration<double> total = end - start;

            if (result.has_value()) {
                correct += compute_recall(ground_truth, result.value()->GetIds(), i, k);
            } else if (result.error().type == vsag::ErrorType::INTERNAL_ERROR) {
                std::cerr << "failed to perform knn search on index" << std::endl;
            }
            if (not(i % 1000)) {
                std::cout << i << std::endl;
            }
            auto search_time = total.count();
            total_time += search_time;
        }
        recall = correct / query_elements;
        double avg_query_time = total_time / query_elements;  // 平均查询时间（秒）
        double qps = query_elements / total_time;             // 每秒查询次数

        std::cout << std::fixed << std::setprecision(6)
                  << "Memory Uasage:" << index->GetMemoryUsage() / 1024.0 / 1024.0 << " MB"
                  << std::endl;
        std::cout << "Correct: " << correct << std::endl;
        std::cout << "elements: " << query_elements << std::endl;
        std::cout << "Recall: " << recall << std::endl;
        std::cout << "Total Query Time (s): " << total_time << std::endl;
        std::cout << "Average Query Time (s): " << avg_query_time << std::endl;
        std::cout << "QPS: " << qps << std::endl;
        std::cout << index->GetStats() << std::endl;
    }
    return 0;
}

int
main() {
    int dim;
    int max_elements;
    int query_elements;
    int max_degree;
    int ef_construction;
    int ef_search;
    int k;
    bool use_static;
    float threshold;
    std::string file_name;
    std::string index_type;

    std::ifstream file("/root/source/vsag/config.json");
    if (!file.is_open()) {
        std::cerr << "Error: Unable to open JSON file." << std::strerror(errno) << std::endl;
        return 1;
    }

    nlohmann::json config;
    file >> config;

    dim = config["dim"].get<int>();
    max_elements = config["max_elements"].get<int>();
    query_elements = config["query_elements"].get<int>();
    max_degree = config["Max_degree"].get<int>();
    ef_construction = config["ef_construction"].get<int>();
    ef_search = config["ef_search"].get<int>();
    k = config["k"].get<int>();
    use_static = config["use_static"].get<bool>();
    threshold = config["threshold"].get<float>();
    file_name = config["file_name"].get<std::string>();
    index_type = config["index_type"].get<std::string>();

    std::cout << "index type: " << index_type << "\n";
    std::cout << "dim: " << dim << std::endl;
    std::cout << "max_elements(M): " << max_elements << std::endl;
    std::cout << "query_elements: " << query_elements << std::endl;
    std::cout << "max_degree: " << max_degree << std::endl;
    std::cout << "ef_construction: " << ef_construction << std::endl;
    std::cout << "ef_search: " << ef_search << std::endl;
    std::cout << "k: " << k << std::endl;
    std::cout << "threshold: " << threshold << std::endl;
    std::cout << "file_name: " << file_name << std::endl;
    nlohmann::json hnsw_parameters{{"max_degree", max_degree},
                                   {"ef_construction", ef_construction},
                                   {"ef_search", ef_search},
                                   {"use_static", use_static}};

    try {
        H5::H5File hdf5_file(file_name.c_str(), H5F_ACC_RDONLY);

        int dimension = 0;

        if (hdf5_file.attrExists("dimension")) {
            H5::Attribute attr = hdf5_file.openAttribute("dimension");
            attr.read(H5::PredType::NATIVE_INT, &dimension);
        } else {
            H5::DataSet train_dataset = hdf5_file.openDataSet("train");
            H5::DataSpace dataspace = train_dataset.getSpace();

            hsize_t dims[2];  // 假设是 2D 数据集
            dataspace.getSimpleExtentDims(dims, nullptr);

            dimension = static_cast<int>(dims[1]);  // 使用第二维度的大小
        }

        std::cout << "Dimension: " << dimension << std::endl;

        H5::DataSet train_dataset = hdf5_file.openDataSet("train");
        H5::DataSpace train_dataspace = train_dataset.getSpace();

        hsize_t train_dims[2];
        train_dataspace.getSimpleExtentDims(train_dims, nullptr);

        std::vector<float> X_train(train_dims[0] * train_dims[1]);
        train_dataset.read(X_train.data(), H5::PredType::NATIVE_FLOAT);

        std::cout << "Train Data Loaded: " << train_dims[0] << " samples, " << train_dims[1]
                  << " features." << std::endl;

        H5::DataSet test_dataset = hdf5_file.openDataSet("test");
        H5::DataSpace test_dataspace = test_dataset.getSpace();

        hsize_t test_dims[2];
        test_dataspace.getSimpleExtentDims(test_dims, nullptr);

        std::vector<float> X_test(train_dims[0] * train_dims[1]);
        test_dataset.read(X_test.data(), H5::PredType::NATIVE_FLOAT);

        std::cout << "Test Data Loaded: " << test_dims[0] << " samples, " << test_dims[1]
                  << " features." << std::endl;

        H5::DataSet ground_truth_dataset = hdf5_file.openDataSet("neighbors");
        H5::DataSpace ground_truth_dataspace = ground_truth_dataset.getSpace();

        hsize_t ground_truth_dims[2];
        ground_truth_dataspace.getSimpleExtentDims(ground_truth_dims, nullptr);

        // 分配线性内存来存储数据
        std::vector<int> linear_data(ground_truth_dims[0] * ground_truth_dims[1]);

        // 读取数据到线性内存中
        ground_truth_dataset.read(linear_data.data(), H5::PredType::NATIVE_INT);

        // 转换为二维 vector
        std::vector<std::vector<int>> ground_truth(ground_truth_dims[0],
                                                   std::vector<int>(ground_truth_dims[1]));
        for (size_t i = 0; i < ground_truth_dims[0]; ++i) {
            for (size_t j = 0; j < ground_truth_dims[1]; ++j) {
                ground_truth[i][j] = linear_data[i * ground_truth_dims[1] + j];
            }
        }

        // 检查前几个 ground_truth 数据
        for (size_t i = 0; i < std::min(ground_truth_dims[0], size_t(5)); ++i) {
            std::cout << "Query " << i << ": ";
            for (size_t j = 0; j < std::min(ground_truth_dims[1], size_t(5)); ++j) {
                std::cout << ground_truth[i][j] << " ";
            }
            std::cout << std::endl;
        }
        // std::vector<std::vector<int>> ground_truth(ground_truth_dims[0], std::vector<int>(ground_truth_dims[1]));
        // ground_truth_dataset.read(ground_truth.at(0).data(), H5::PredType::NATIVE_INT);

        std::cout << "Ground Truth Data Loaded: " << ground_truth_dims[0] << " samples, "
                  << ground_truth_dims[1] << " features." << std::endl;

        bench(index_type,
              max_elements,
              query_elements,
              dim,
              ef_search,
              k,
              hnsw_parameters,
              X_train,
              X_test,
              ground_truth);
    } catch (const FileIException& e) {
        e.printErrorStack();
    } catch (const DataSetIException& e) {
        e.printErrorStack();
    } catch (const DataSpaceIException& e) {
        e.printErrorStack();
    } catch (std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }

    return 0;
}
