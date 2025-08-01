/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/Index.h>
#include <typeinfo>
#include <cxxabi.h>
#include <iostream>
#include <map>
#include <unordered_map>
#include <cstdlib>

namespace faiss {
// 对齐分配器
template<typename T, std::size_t Alignment>
struct AlignedAllocator {
    using value_type = T;

    template<typename U>
    struct rebind { using other = AlignedAllocator<U, Alignment>; };

    AlignedAllocator() noexcept = default;
    template<typename U>
    AlignedAllocator(const AlignedAllocator<U, Alignment>&) noexcept {}

    T* allocate(std::size_t n) {
        // C++17 对齐分配
        void* ptr = ::operator new(n * sizeof(T), std::align_val_t(Alignment));
        return static_cast<T*>(ptr);
    }

    void deallocate(T* p, std::size_t) noexcept {
        ::operator delete(p, std::align_val_t(Alignment));
    }

    bool operator==(const AlignedAllocator&) const noexcept { return true; }
    bool operator!=(const AlignedAllocator&) const noexcept { return false; }
};

// typedef std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> AlignedBF16Vec;
// typedef std::unordered_map<size_t, AlignedBF16Vec> BF16Cache;

typedef std::vector<uint16_t> AlignedBF16Vec;
typedef std::unordered_map<size_t, AlignedBF16Vec> BF16Cache;

/***********************************************************
 * The distance computer maintains a current query and computes
 * distances to elements in an index that supports random access.
 *
 * The DistanceComputer is not intended to be thread-safe (eg. because
 * it maintains counters) so the distance functions are not const,
 * instantiate one from each thread if needed.
 *
 * Note that the equivalent for IVF indexes is the InvertedListScanner,
 * that has additional methods to handle the inverted list context.
 ***********************************************************/
struct DistanceComputer {
    /// called before computing distances. Pointer x should remain valid
    /// while operator () is called
    virtual void set_query(const float* x) = 0;

    /// compute distance of vector i to current query
    virtual float operator()(idx_t i) = 0;

    /// compute distances of current query to 4 stored vectors.
    /// certain DistanceComputer implementations may benefit
    /// heavily from this.
    virtual void distances_batch_4(
            const idx_t idx0,
            const idx_t idx1,
            const idx_t idx2,
            const idx_t idx3,
            float& dis0,
            float& dis1,
            float& dis2,
            float& dis3) {
        // compute first, assign next
        const float d0 = this->operator()(idx0);
        const float d1 = this->operator()(idx1);
        const float d2 = this->operator()(idx2);
        const float d3 = this->operator()(idx3);
        dis0 = d0;
        dis1 = d1;
        dis2 = d2;
        dis3 = d3;
    }

    virtual void distances_batch(
            const size_t* idx,
            float* dis,
            size_t stride, 
            BF16Cache* visited_bf_vec) {
        for (size_t i = 0; i < stride; i++) {
            dis[i] = this->operator()(idx[i]);
        }
    }    

    /// compute distance between two stored vectors
    virtual float symmetric_dis(idx_t i, idx_t j) = 0;

    virtual ~DistanceComputer() {}
};

/* Wrap the distance computer into one that negates the
   distances. This makes supporting INNER_PRODUCE search easier */

struct NegativeDistanceComputer : DistanceComputer {
    /// owned by this
    DistanceComputer* basedis;

    explicit NegativeDistanceComputer(DistanceComputer* basedis)
            : basedis(basedis) {}

    void set_query(const float* x) override {
        basedis->set_query(x);
    }

    /// compute distance of vector i to current query
    float operator()(idx_t i) override {
        return -(*basedis)(i);
    }

    void distances_batch_4(
            const idx_t idx0,
            const idx_t idx1,
            const idx_t idx2,
            const idx_t idx3,
            float& dis0,
            float& dis1,
            float& dis2,
            float& dis3) override {
        basedis->distances_batch_4(
                idx0, idx1, idx2, idx3, dis0, dis1, dis2, dis3);
        dis0 = -dis0;
        dis1 = -dis1;
        dis2 = -dis2;
        dis3 = -dis3;
    }

    void distances_batch(
            const size_t* idx,
            float* dis,  size_t stride, BF16Cache* visited_bf_vec) override {
        basedis->distances_batch(idx, dis, stride, visited_bf_vec);
    }      

    /// compute distance between two stored vectors
    float symmetric_dis(idx_t i, idx_t j) override {
        return -basedis->symmetric_dis(i, j);
    }

    virtual ~NegativeDistanceComputer() {
        delete basedis;
    }
};

/*************************************************************
 * Specialized version of the DistanceComputer when we know that codes are
 * laid out in a flat index.
 */
struct FlatCodesDistanceComputer : DistanceComputer {
    const uint8_t* codes;
    size_t code_size;

    FlatCodesDistanceComputer(const uint8_t* codes, size_t code_size)
            : codes(codes), code_size(code_size) {}

    FlatCodesDistanceComputer() : codes(nullptr), code_size(0) {}

    float operator()(idx_t i) override {
        return distance_to_code(codes + i * code_size);
    }

    /// compute distance of current query to an encoded vector
    virtual float distance_to_code(const uint8_t* code) = 0;

    virtual ~FlatCodesDistanceComputer() {}
};

} // namespace faiss
