#pragma once

#include <map>
#include <memory>
#include <shared_mutex>

#include "../utils.h"
#include "algorithm/hnswlib/algorithm_interface.h"
#include "algorithm/hnswlib/hnswalg.h"
#include "algorithm/hnswlib/space_interface.h"
#include "data_cell/glass_graph.hpp"
#include "data_type.h"
#include "glass_searcher.hpp"
#include "impl/conjugate_graph.h"
#include "safe_allocator.h"
#include "vsag/index.h"

namespace vsag {

class Glass : public Index {
public:
    Glass(std::shared_ptr<hnswlib::SpaceInterface> space_interface,
          int M,
          int ef_construction,
          DataTypes type,
          bool use_static = false,
          bool use_reversed_edges = false,
          bool use_conjugate_graph = false,
          bool normalize = false,
          Allocator* allocator = nullptr);

    virtual ~Glass() {
        alg_hnsw_ = nullptr;
        allocator_.reset();
    }

public:
    tl::expected<std::vector<int64_t>, Error>
    Build(const DatasetPtr& base) override {
        SAFE_CALL(return this->build(base));
    }

    tl::expected<std::vector<int64_t>, Error>
    Add(const DatasetPtr& base) override {
        SAFE_CALL(return this->add(base));
    }

    tl::expected<bool, Error>
    Remove(int64_t id) override {
        SAFE_CALL(return this->remove(id));
    }

    tl::expected<DatasetPtr, Error>
    KnnSearch(const DatasetPtr& query,
              int64_t k,
              const std::string& parameters,
              const std::function<bool(int64_t)>& filter) const override {
        SAFE_CALL(return this->knn_search_internal(query, k, parameters, filter));
    }

    tl::expected<DatasetPtr, Error>
    KnnSearch(const DatasetPtr& query,
              int64_t k,
              const std::string& parameters,
              BitsetPtr invalid = nullptr) const override {
        SAFE_CALL(return this->knn_search_internal(query, k, parameters, invalid));
    }

    tl::expected<DatasetPtr, Error>
    RangeSearch(const DatasetPtr& query,
                float radius,
                const std::string& parameters,
                int64_t limited_size = -1) const override {
        SAFE_CALL(return this->range_search_internal(
            query, radius, parameters, (BitsetPtr) nullptr, limited_size));
    }

    tl::expected<DatasetPtr, Error>
    RangeSearch(const DatasetPtr& query,
                float radius,
                const std::string& parameters,
                const std::function<bool(int64_t)>& filter,
                int64_t limited_size = -1) const override {
        SAFE_CALL(
            return this->range_search_internal(query, radius, parameters, filter, limited_size));
    }

    tl::expected<DatasetPtr, Error>
    RangeSearch(const DatasetPtr& query,
                float radius,
                const std::string& parameters,
                BitsetPtr invalid,
                int64_t limited_size = -1) const override {
        SAFE_CALL(
            return this->range_search_internal(query, radius, parameters, invalid, limited_size));
    }

    tl::expected<uint32_t, Error>
    Feedback(const DatasetPtr& query,
             int64_t k,
             const std::string& parameters,
             int64_t global_optimum_tag_id = std::numeric_limits<int64_t>::max()) override {
        SAFE_CALL(return this->feedback(query, k, parameters, global_optimum_tag_id));
    };

    tl::expected<uint32_t, Error>
    Pretrain(const std::vector<int64_t>& base_tag_ids,
             uint32_t k,
             const std::string& parameters) override {
        SAFE_CALL(return this->pretrain(base_tag_ids, k, parameters));
    };

    virtual tl::expected<float, Error>
    CalcDistanceById(const float* vector, int64_t id) const override {
        SAFE_CALL(return alg_hnsw_->getDistanceByLabel(id, vector));
    };

public:
    tl::expected<BinarySet, Error>
    Serialize() const override {
        SAFE_CALL(return this->serialize());
    }

    tl::expected<void, Error>
    Serialize(std::ostream& out_stream) override {
        SAFE_CALL(return this->serialize(out_stream));
    }

    tl::expected<void, Error>
    Deserialize(const BinarySet& binary_set) override {
        SAFE_CALL(return this->deserialize(binary_set));
    }

    tl::expected<void, Error>
    Deserialize(const ReaderSet& reader_set) override {
        SAFE_CALL(return this->deserialize(reader_set));
    }

    tl::expected<void, Error>
    Deserialize(std::istream& in_stream) override {
        SAFE_CALL(return this->deserialize(in_stream));
    }

public:
    int64_t
    GetNumElements() const override {
        return alg_hnsw_->getCurrentElementCount() - alg_hnsw_->getDeletedCount();
    }

    int64_t
    GetMemoryUsage() const override {
        if (use_conjugate_graph_)
            return alg_hnsw_->calcSerializeSize() + conjugate_graph_->GetMemoryUsage();
        else
            return alg_hnsw_->calcSerializeSize();
    }

    std::string
    GetStats() const override;

    // used to test the integrity of graphs, used only in UT.
    bool
    CheckGraphIntegrity() const;

private:
    tl::expected<std::vector<int64_t>, Error>
    build(const DatasetPtr& base);

    tl::expected<std::vector<int64_t>, Error>
    add(const DatasetPtr& base);

    tl::expected<bool, Error>
    remove(int64_t id);

    template <typename FilterType>
    tl::expected<DatasetPtr, Error>
    knn_search_internal(const DatasetPtr& query,
                        int64_t k,
                        const std::string& parameters,
                        const FilterType& filter_obj) const;

    tl::expected<DatasetPtr, Error>
    knn_search(const DatasetPtr& query,
               int64_t k,
               const std::string& parameters,
               BaseFilterFunctor* filter_ptr) const;

    template <typename FilterType>
    tl::expected<DatasetPtr, Error>
    range_search_internal(const DatasetPtr& query,
                          float radius,
                          const std::string& parameters,
                          const FilterType& filter_obj,
                          int64_t limited_size) const;

    tl::expected<DatasetPtr, Error>
    range_search(const DatasetPtr& query,
                 float radius,
                 const std::string& parameters,
                 BaseFilterFunctor* filter_ptr,
                 int64_t limited_size) const;

    tl::expected<uint32_t, Error>
    feedback(const DatasetPtr& query,
             int64_t k,
             const std::string& parameters,
             int64_t global_optimum_tag_id);

    tl::expected<uint32_t, Error>
    feedback(const DatasetPtr& result, int64_t global_optimum_tag_id, int64_t k);

    tl::expected<DatasetPtr, Error>
    brute_force(const DatasetPtr& query, int64_t k);

    tl::expected<uint32_t, Error>
    pretrain(const std::vector<int64_t>& base_tag_ids, uint32_t k, const std::string& parameters);

    tl::expected<BinarySet, Error>
    serialize() const;

    tl::expected<void, Error>
    serialize(std::ostream& out_stream);

    tl::expected<void, Error>
    deserialize(const BinarySet& binary_set);

    tl::expected<void, Error>
    deserialize(const ReaderSet& binary_set);

    tl::expected<void, Error>
    deserialize(std::istream& in_stream);

    tl::expected<bool, Error>
    init_memory_space();

    void
    get_vectors(const DatasetPtr& base, void** vectors_ptr, size_t* data_size_ptr) const;

    BinarySet
    empty_binaryset() const;

private:
    std::shared_ptr<hnswlib::HierarchicalNSW> alg_hnsw_;
    std::shared_ptr<hnswlib::SpaceInterface> space_;

    bool use_conjugate_graph_;
    std::shared_ptr<ConjugateGraph> conjugate_graph_;

    int64_t dim_;
    bool use_static_ = false;
    bool empty_index_ = false;
    bool use_reversed_edges_ = false;
    bool is_init_memory_ = false;
    DataTypes type_;

    std::shared_ptr<SafeAllocator> allocator_;

    mutable std::mutex stats_mutex_;
    mutable std::map<std::string, WindowResultQueue> result_queues_;

    mutable std::shared_mutex rw_mutex_;

    uint32_t M_;
    glass::Graph<int> final_graph_;
    std::unique_ptr<glass::SearcherBase> searcher_;
};

}  // namespace vsag