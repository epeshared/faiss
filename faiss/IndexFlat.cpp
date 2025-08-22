/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#include <faiss/IndexFlat.h>

#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/DistanceComputer.h>
#include <faiss/utils/Heap.h>
#include <faiss/utils/distances.h>
#include <faiss/utils/extra_distances.h>
#include <faiss/utils/prefetch.h>
#include <faiss/utils/sorting.h>
#include <cstring>
#include <immintrin.h>
#include <map>
#include <unordered_map>
#include <cstdint>
#include <cassert>
#if defined(ENABLE_AMX)
// #include <faiss/utils/onednn_utils.h>
// #include <faiss/utils/amx_utils.h>
#include <faiss/utils/amx_bf16.h>
#include <faiss/utils/amx_bf16_block.h>
#endif

namespace faiss {

IndexFlat::IndexFlat(idx_t d, MetricType metric)
        : IndexFlatCodes(sizeof(float) * d, d, metric) {}

void IndexFlat::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    IDSelector* sel = params ? params->sel : nullptr;
    FAISS_THROW_IF_NOT(k > 0);

    // we see the distances and labels as heaps
    if (metric_type == METRIC_INNER_PRODUCT) {
        float_minheap_array_t res = {size_t(n), size_t(k), labels, distances};
        knn_inner_product(x, get_xb(), d, n, ntotal, &res, sel);
    } else if (metric_type == METRIC_L2) {
        float_maxheap_array_t res = {size_t(n), size_t(k), labels, distances};
        knn_L2sqr(x, get_xb(), d, n, ntotal, &res, nullptr, sel);
    } else {
        FAISS_THROW_IF_NOT(!sel); // TODO implement with selector
        knn_extra_metrics(
                x,
                get_xb(),
                d,
                n,
                ntotal,
                metric_type,
                metric_arg,
                k,
                distances,
                labels);
    }
}

void IndexFlat::range_search(
        idx_t n,
        const float* x,
        float radius,
        RangeSearchResult* result,
        const SearchParameters* params) const {
    IDSelector* sel = params ? params->sel : nullptr;

    switch (metric_type) {
        case METRIC_INNER_PRODUCT:
            range_search_inner_product(
                    x, get_xb(), d, n, ntotal, radius, result, sel);
            break;
        case METRIC_L2:
            range_search_L2sqr(x, get_xb(), d, n, ntotal, radius, result, sel);
            break;
        default:
            FAISS_THROW_MSG("metric type not supported");
    }
}

void IndexFlat::compute_distance_subset(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        const idx_t* labels) const {
    switch (metric_type) {
        case METRIC_INNER_PRODUCT:
            fvec_inner_products_by_idx(distances, x, get_xb(), labels, d, n, k);
            break;
        case METRIC_L2:
            fvec_L2sqr_by_idx(distances, x, get_xb(), labels, d, n, k);
            break;
        default:
            FAISS_THROW_MSG("metric type not supported");
    }
}

namespace {

struct FlatL2Dis : FlatCodesDistanceComputer {
    size_t d;
    idx_t nb;
    const float* q;
    const float* b;
    size_t ndis;

    float distance_to_code(const uint8_t* code) final {
        ndis++;
        return fvec_L2sqr(q, (float*)code, d);
    }

    float symmetric_dis(idx_t i, idx_t j) override {
        return fvec_L2sqr(b + j * d, b + i * d, d);
    }

    explicit FlatL2Dis(const IndexFlat& storage, const float* q = nullptr)
            : FlatCodesDistanceComputer(
                      storage.codes.data(),
                      storage.code_size),
              d(storage.d),
              nb(storage.ntotal),
              q(q),
              b(storage.get_xb()),
              ndis(0) {}

    void set_query(const float* x) override {
        q = x;
    }

    // compute four distances
    void distances_batch_4(
            const idx_t idx0,
            const idx_t idx1,
            const idx_t idx2,
            const idx_t idx3,
            float& dis0,
            float& dis1,
            float& dis2,
            float& dis3) final override {
        ndis += 4;

        // compute first, assign next
        const float* __restrict y0 =
                reinterpret_cast<const float*>(codes + idx0 * code_size);
        const float* __restrict y1 =
                reinterpret_cast<const float*>(codes + idx1 * code_size);
        const float* __restrict y2 =
                reinterpret_cast<const float*>(codes + idx2 * code_size);
        const float* __restrict y3 =
                reinterpret_cast<const float*>(codes + idx3 * code_size);

        float dp0 = 0;
        float dp1 = 0;
        float dp2 = 0;
        float dp3 = 0;
        fvec_L2sqr_batch_4(q, y0, y1, y2, y3, d, dp0, dp1, dp2, dp3);
        dis0 = dp0;
        dis1 = dp1;
        dis2 = dp2;
        dis3 = dp3;
    }
};

struct FlatIPDis : FlatCodesDistanceComputer {
    size_t d;
    idx_t nb;
    const float* q;
    const float* b;
    size_t ndis;

    float symmetric_dis(idx_t i, idx_t j) final override {
        return fvec_inner_product(b + j * d, b + i * d, d);
    }

    float distance_to_code(const uint8_t* code) final override {
        ndis++;
        return fvec_inner_product(q, (const float*)code, d);
    }

    explicit FlatIPDis(const IndexFlat& storage, const float* q = nullptr)
            : FlatCodesDistanceComputer(
                      storage.codes.data(),
                      storage.code_size),
              d(storage.d),
              nb(storage.ntotal),
              q(q),
              b(storage.get_xb()),
              ndis(0) {}

    void set_query(const float* x) override {
        q = x;
    }

    // compute four distances
    void distances_batch_4(
            const idx_t idx0,
            const idx_t idx1,
            const idx_t idx2,
            const idx_t idx3,
            float& dis0,
            float& dis1,
            float& dis2,
            float& dis3) final override {
        ndis += 4;

        // compute first, assign next
        const float* __restrict y0 =
                reinterpret_cast<const float*>(codes + idx0 * code_size);
        const float* __restrict y1 =
                reinterpret_cast<const float*>(codes + idx1 * code_size);
        const float* __restrict y2 =
                reinterpret_cast<const float*>(codes + idx2 * code_size);
        const float* __restrict y3 =
                reinterpret_cast<const float*>(codes + idx3 * code_size);

        float dp0 = 0;
        float dp1 = 0;
        float dp2 = 0;
        float dp3 = 0;
        fvec_inner_product_batch_4(q, y0, y1, y2, y3, d, dp0, dp1, dp2, dp3);
        dis0 = dp0;
        dis1 = dp1;
        dis2 = dp2;
        dis3 = dp3;
    }

#if defined(__AVX512BF16__)

    /**
     * @param src_base      pointer to float data laid out in rows of byte-size `code_size`
     * @param dst           array of `batch_size` pointers; on return dst[i] points at the bf16 row
     * @param idx           optional array of row‑indices; pass nullptr to use i as the key
     * @param batch_size    number of rows to convert
     * @param code_size     bytes per row of float data (usually dim * sizeof(float))
     * @param dim           number of floats per row
     * @param cache         optional cache of previously converted rows; pass nullptr to disable
     */
    static inline void convert_f32_to_bf16_avx512bf16(
        const void*    src_base,     // 连续的按行存放的 float32 数据
        uint16_t**     dst,          // 输出指针数组：dst[i] 指向第 i 个样本的 bf16 行缓冲
        const size_t*  idx,          // 行索引（可为空；为空则用 i）
        size_t         batch_size,   // 处理的行数
        size_t         code_size,    // 每行步幅（字节）
        size_t         dim          // 每行有效元素个数
        )  // 可选缓存；为 nullptr 时不查不写
    {        
        assert(src_base != nullptr);
        assert(dst != nullptr);
        assert(code_size % sizeof(float) == 0);

        const float* float_base = reinterpret_cast<const float*>(src_base);
        const size_t floats_per_row = code_size / sizeof(float);

        // 若存在 padding，确保源行至少有 dim 个有效元素
        assert(floats_per_row >= dim || dim == 0);

        const size_t tail = dim & 15u; // dim % 16
        const __mmask16 tail_mask = tail ? ((__mmask16(1) << tail) - 1) : (__mmask16)0;
        const size_t main = dim - tail;

        for (size_t i = 0; i < batch_size; ++i) {
            const size_t key = idx ? idx[i] : i;
            const float* src_row = float_base + key * floats_per_row;

            uint16_t* out_ptr = dst[i];
            assert(out_ptr != nullptr); // 需要调用方保证有 dim 个 u16 可写空间

            // 向量化主循环：每次 16 个
            size_t j = 0;
            for (; j < main; j += 16) {
                __m512    v  = _mm512_loadu_ps(src_row + j);
                __m256bh  ph = _mm512_cvtneps_pbh(v); // f32 -> bf16 (RNNE)
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(out_ptr + j), (__m256i)ph);
            }
            
            // 尾部（<16）
            if (tail) {
                __m512    v_tail  = _mm512_maskz_loadu_ps(tail_mask, src_row + j);
                __m256bh  ph_tail = _mm512_cvtneps_pbh(v_tail);
                _mm256_mask_storeu_epi16(reinterpret_cast<__m256i*>(out_ptr + j),
                                        tail_mask, (__m256i)ph_tail);
            }
            // 记录输出指针
            dst[i] = out_ptr;
        }
    }

#endif    

    void distances_batch(
            const size_t* idx,
            float* dis, size_t stride) final override {
#if  defined(ENABLE_AMX)  && defined(__AVX512F__)
        ndis += stride;
        
        uint16_t* b_ptrs[stride];
        auto *raw = (uint16_t*) aligned_alloc(64, stride * d * sizeof(uint16_t));
        std::unique_ptr<uint16_t, void(*)(void*)> b_pool(raw, free);

        for (size_t i = 0; i < stride; ++i) b_ptrs[i] = raw + i * d;

        convert_f32_to_bf16_avx512bf16(
            codes,
            b_ptrs,
            idx,
            stride,
            code_size,
            d);
        
        PackedA A; 
        prepack_A_bf16_tiles((void **)b_ptrs, stride, d, A);

        
        uint16_t* q_ptrs[1] = { nullptr };
        auto *q_raw = (uint16_t*) aligned_alloc(64, 1 * d * sizeof(uint16_t));
        std::unique_ptr<uint16_t, void(*)(void*)> q_pool(q_raw, free);
        q_ptrs[0] = q_raw;
        convert_f32_to_bf16_avx512bf16(
            q,
            q_ptrs,
            0,
            1,
            code_size,
            d);
        
        amx_row_scores_packed(b_ptrs, q_ptrs[1], d, stride, 1, A, dis);
        free_packedA(A);

#else
        for (size_t i = 0; i < 16; i++) {
            dis[i] = fvec_inner_product(
                    q,
                    reinterpret_cast<const float*>(codes + idx[i] * code_size),
                    d);
        }
#endif        
    }

};

} // namespace

FlatCodesDistanceComputer* IndexFlat::get_FlatCodesDistanceComputer() const {
    if (metric_type == METRIC_L2) {
        return new FlatL2Dis(*this);
    } else if (metric_type == METRIC_INNER_PRODUCT) {
        return new FlatIPDis(*this);
    } else {
        return get_extra_distance_computer(
                d, metric_type, metric_arg, ntotal, get_xb());
    }
}

void IndexFlat::reconstruct(idx_t key, float* recons) const {
    FAISS_THROW_IF_NOT(key < ntotal);
    memcpy(recons, &(codes[key * code_size]), code_size);
}

void IndexFlat::sa_encode(idx_t n, const float* x, uint8_t* bytes) const {
    if (n > 0) {
        memcpy(bytes, x, sizeof(float) * d * n);
    }
}

void IndexFlat::sa_decode(idx_t n, const uint8_t* bytes, float* x) const {
    if (n > 0) {
        memcpy(x, bytes, sizeof(float) * d * n);
    }
}

/***************************************************
 * IndexFlatL2
 ***************************************************/

namespace {
struct FlatL2WithNormsDis : FlatCodesDistanceComputer {
    size_t d;
    idx_t nb;
    const float* q;
    const float* b;
    size_t ndis;

    const float* l2norms;
    float query_l2norm;

    float distance_to_code(const uint8_t* code) final override {
        ndis++;
        return fvec_L2sqr(q, (float*)code, d);
    }

    float operator()(const idx_t i) final override {
        const float* __restrict y =
                reinterpret_cast<const float*>(codes + i * code_size);

        prefetch_L2(l2norms + i);
        const float dp0 = fvec_inner_product(q, y, d);
        return query_l2norm + l2norms[i] - 2 * dp0;
    }

    float symmetric_dis(idx_t i, idx_t j) final override {
        const float* __restrict yi =
                reinterpret_cast<const float*>(codes + i * code_size);
        const float* __restrict yj =
                reinterpret_cast<const float*>(codes + j * code_size);

        prefetch_L2(l2norms + i);
        prefetch_L2(l2norms + j);
        const float dp0 = fvec_inner_product(yi, yj, d);
        return l2norms[i] + l2norms[j] - 2 * dp0;
    }

    explicit FlatL2WithNormsDis(
            const IndexFlatL2& storage,
            const float* q = nullptr)
            : FlatCodesDistanceComputer(
                      storage.codes.data(),
                      storage.code_size),
              d(storage.d),
              nb(storage.ntotal),
              q(q),
              b(storage.get_xb()),
              ndis(0),
              l2norms(storage.cached_l2norms.data()),
              query_l2norm(0) {}

    void set_query(const float* x) override {
        q = x;
        query_l2norm = fvec_norm_L2sqr(q, d);
    }

    // compute four distances
    void distances_batch_4(
            const idx_t idx0,
            const idx_t idx1,
            const idx_t idx2,
            const idx_t idx3,
            float& dis0,
            float& dis1,
            float& dis2,
            float& dis3) final override {
        ndis += 4;

        // compute first, assign next
        const float* __restrict y0 =
                reinterpret_cast<const float*>(codes + idx0 * code_size);
        const float* __restrict y1 =
                reinterpret_cast<const float*>(codes + idx1 * code_size);
        const float* __restrict y2 =
                reinterpret_cast<const float*>(codes + idx2 * code_size);
        const float* __restrict y3 =
                reinterpret_cast<const float*>(codes + idx3 * code_size);

        prefetch_L2(l2norms + idx0);
        prefetch_L2(l2norms + idx1);
        prefetch_L2(l2norms + idx2);
        prefetch_L2(l2norms + idx3);

        float dp0 = 0;
        float dp1 = 0;
        float dp2 = 0;
        float dp3 = 0;
        fvec_inner_product_batch_4(q, y0, y1, y2, y3, d, dp0, dp1, dp2, dp3);
        dis0 = query_l2norm + l2norms[idx0] - 2 * dp0;
        dis1 = query_l2norm + l2norms[idx1] - 2 * dp1;
        dis2 = query_l2norm + l2norms[idx2] - 2 * dp2;
        dis3 = query_l2norm + l2norms[idx3] - 2 * dp3;
    }
};

} // namespace

void IndexFlatL2::sync_l2norms() {
    cached_l2norms.resize(ntotal);
    fvec_norms_L2sqr(
            cached_l2norms.data(),
            reinterpret_cast<const float*>(codes.data()),
            d,
            ntotal);
}

void IndexFlatL2::clear_l2norms() {
    cached_l2norms.clear();
    cached_l2norms.shrink_to_fit();
}

FlatCodesDistanceComputer* IndexFlatL2::get_FlatCodesDistanceComputer() const {
    if (metric_type == METRIC_L2) {
        if (!cached_l2norms.empty()) {
            return new FlatL2WithNormsDis(*this);
        }
    }

    return IndexFlat::get_FlatCodesDistanceComputer();
}

/***************************************************
 * IndexFlat1D
 ***************************************************/

IndexFlat1D::IndexFlat1D(bool continuous_update)
        : IndexFlatL2(1), continuous_update(continuous_update) {}

/// if not continuous_update, call this between the last add and
/// the first search
void IndexFlat1D::update_permutation() {
    perm.resize(ntotal);
    if (ntotal < 1000000) {
        fvec_argsort(ntotal, get_xb(), (size_t*)perm.data());
    } else {
        fvec_argsort_parallel(ntotal, get_xb(), (size_t*)perm.data());
    }
}

void IndexFlat1D::add(idx_t n, const float* x) {
    IndexFlatL2::add(n, x);
    if (continuous_update)
        update_permutation();
}

void IndexFlat1D::reset() {
    IndexFlatL2::reset();
    perm.clear();
}

void IndexFlat1D::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    FAISS_THROW_IF_NOT_MSG(
            !params, "search params not supported for this index");
    FAISS_THROW_IF_NOT(k > 0);
    FAISS_THROW_IF_NOT_MSG(
            perm.size() == ntotal, "Call update_permutation before search");
    const float* xb = get_xb();

#pragma omp parallel for if (n > 10000)
    for (idx_t i = 0; i < n; i++) {
        float q = x[i]; // query
        float* D = distances + i * k;
        idx_t* I = labels + i * k;

        // binary search
        idx_t i0 = 0, i1 = ntotal;
        idx_t wp = 0;

        if (ntotal == 0) {
            for (idx_t j = 0; j < k; j++) {
                I[j] = -1;
                D[j] = HUGE_VAL;
            }
            goto done;
        }

        if (xb[perm[i0]] > q) {
            i1 = 0;
            goto finish_right;
        }

        if (xb[perm[i1 - 1]] <= q) {
            i0 = i1 - 1;
            goto finish_left;
        }

        while (i0 + 1 < i1) {
            idx_t imed = (i0 + i1) / 2;
            if (xb[perm[imed]] <= q)
                i0 = imed;
            else
                i1 = imed;
        }

        // query is between xb[perm[i0]] and xb[perm[i1]]
        // expand to nearest neighs

        while (wp < k) {
            float xleft = xb[perm[i0]];
            float xright = xb[perm[i1]];

            if (q - xleft < xright - q) {
                D[wp] = q - xleft;
                I[wp] = perm[i0];
                i0--;
                wp++;
                if (i0 < 0) {
                    goto finish_right;
                }
            } else {
                D[wp] = xright - q;
                I[wp] = perm[i1];
                i1++;
                wp++;
                if (i1 >= ntotal) {
                    goto finish_left;
                }
            }
        }
        goto done;

    finish_right:
        // grow to the right from i1
        while (wp < k) {
            if (i1 < ntotal) {
                D[wp] = xb[perm[i1]] - q;
                I[wp] = perm[i1];
                i1++;
            } else {
                D[wp] = std::numeric_limits<float>::infinity();
                I[wp] = -1;
            }
            wp++;
        }
        goto done;

    finish_left:
        // grow to the left from i0
        while (wp < k) {
            if (i0 >= 0) {
                D[wp] = q - xb[perm[i0]];
                I[wp] = perm[i0];
                i0--;
            } else {
                D[wp] = std::numeric_limits<float>::infinity();
                I[wp] = -1;
            }
            wp++;
        }
    done:;
    }
}

} // namespace faiss

