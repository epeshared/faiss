/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cinttypes>
#include <cstring>
#include <execinfo.h>
#include <limits>

#include <faiss/impl/ScalarQuantizer.h>
#include <faiss/utils/AlignedTable.h>
#include <faiss/utils/simd_levels.h>
#include <faiss/utils/simdlib.h>
#include <faiss/utils/bf16.h>
#include <faiss/utils/amx_tile_bf16.h>

namespace faiss {

namespace scalar_quantizer {

using SQDistanceComputer = ScalarQuantizer::SQDistanceComputer;

/*******************************************************************
 * DistanceComputer: combines a similarity and a quantizer to do
 * code-to-vector or code-to-code comparisons
 *******************************************************************/

template <class Quantizer, class Similarity, SIMDLevel SL>
struct DCTemplate : SQDistanceComputer {};

#if defined(__AVX512BF16__)

// Fast path for QT_bf16 + IP on CPUs with AVX512_BF16.
//
// Key idea: quantize query to BF16 once in set_query(), then compute inner
// products using VDPBF16PS against BF16-coded vectors.
//
// Notes:
// - Only enabled when __AVX512BF16__ is available (e.g., -march=sapphirerapids).
// - Uses dpbf16 over 32-element blocks and falls back to scalar BF16 for the
//   tail when d % 32 != 0 (e.g., d = 1136).
template <SIMDLevel SL>
struct DCBF16IPDpbf16 : SQDistanceComputer {
    using Sim = SimilarityIP<SL>;

    QuantizerBF16<SL> quant;
    std::vector<uint16_t> qbf16;

    DCBF16IPDpbf16(size_t d, const std::vector<float>& trained)
            : quant(d, trained), qbf16(d) {}

    void set_query(const float* x) final {
        q = x;
        // Match QuantizerBF16::encode_vector semantics (encode_bf16()).
        for (size_t i = 0; i < quant.d; i++) {
            qbf16[i] = encode_bf16(x[i]);
        }
    }

    FAISS_ALWAYS_INLINE float compute_code_ip_bf16(
            const uint16_t* a,
            const uint16_t* b) const {
        // Use dpbf16 over as many 32-element chunks as possible.
        __m512 acc = _mm512_setzero_ps();
        const size_t d32 = quant.d & ~size_t(31);
        for (size_t i = 0; i < d32; i += 32) {
            const __m512i va = _mm512_loadu_si512((const void*)(a + i));
            const __m512i vb = _mm512_loadu_si512((const void*)(b + i));
            const __m512bh bha = (__m512bh)va;
            const __m512bh bhb = (__m512bh)vb;
            acc = _mm512_dpbf16_ps(acc, bha, bhb);
        }
        float res = _mm512_reduce_add_ps(acc);
        for (size_t i = d32; i < quant.d; i++) {
            res += decode_bf16(a[i]) * decode_bf16(b[i]);
        }
        return res;
    }

    float partial_dot_product(
            const idx_t i,
            const uint32_t offset,
            const uint32_t num_components) final override {
        if (i < 0) {
            FAISS_THROW_FMT(
                    "partial_dot_product called with negative id=%" PRId64,
                    (int64_t)i);
        }
        if (codes == nullptr) {
            FAISS_THROW_MSG("partial_dot_product called with null codes");
        }
        if (code_size != quant.d * sizeof(uint16_t)) {
            FAISS_THROW_FMT(
                    "partial_dot_product unexpected code_size=%zu (expected %zu)",
                    code_size,
                    quant.d * sizeof(uint16_t));
        }

        if (std::getenv("FAISS_DEBUG_PARTIAL_DOT") != nullptr) {
            fprintf(
                    stderr,
                    "[faiss] partial_dot_product: id=%" PRId64
                    " offset=%u num=%u d=%zu code_size=%zu codes=%p\n",
                    (int64_t)i,
                    offset,
                    num_components,
                    quant.d,
                    code_size,
                    (const void*)codes);
            void* bt[48];
            const int nbt = backtrace(bt, 48);
            backtrace_symbols_fd(bt, nbt, fileno(stderr));
            fflush(stderr);
        }

        const uint32_t start = offset;
        const uint32_t end =
                std::min<uint32_t>(uint32_t(quant.d), start + num_components);
        if (start >= end) {
            return 0.0f;
        }

        const auto* qv = qbf16.data();
        const size_t ii = (size_t)i;
        if (ii > (std::numeric_limits<size_t>::max() / code_size)) {
            FAISS_THROW_MSG("partial_dot_product id overflow");
        }
        const auto* c = (const uint16_t*)(codes + ii * code_size);

        float res = 0.0f;
        uint32_t j = start;

        // Scalar head until 32-aligned.
        for (; j < end && (j & 31u) != 0; j++) {
            res += decode_bf16(qv[j]) * decode_bf16(c[j]);
        }

        // Vectorized body.
        __m512 acc = _mm512_setzero_ps();
        const uint32_t end32 = end & ~31u;
        for (; j < end32; j += 32) {
            const __m512i vq = _mm512_loadu_si512((const void*)(qv + j));
            const __m512i vc = _mm512_loadu_si512((const void*)(c + j));
            acc = _mm512_dpbf16_ps(acc, (__m512bh)vq, (__m512bh)vc);
        }
        res += _mm512_reduce_add_ps(acc);

        // Scalar tail.
        for (; j < end; j++) {
            res += decode_bf16(qv[j]) * decode_bf16(c[j]);
        }

        return res;
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
        const auto* qv = qbf16.data();
        const auto* c0 = (const uint16_t*)(codes + idx0 * code_size);
        const auto* c1 = (const uint16_t*)(codes + idx1 * code_size);
        const auto* c2 = (const uint16_t*)(codes + idx2 * code_size);
        const auto* c3 = (const uint16_t*)(codes + idx3 * code_size);

        __m512 acc0 = _mm512_setzero_ps();
        __m512 acc1 = _mm512_setzero_ps();
        __m512 acc2 = _mm512_setzero_ps();
        __m512 acc3 = _mm512_setzero_ps();

        const size_t d32 = quant.d & ~size_t(31);
        for (size_t i = 0; i < d32; i += 32) {
            const __m512i vq = _mm512_loadu_si512((const void*)(qv + i));
            const __m512bh bq = (__m512bh)vq;

            const __m512i v0 = _mm512_loadu_si512((const void*)(c0 + i));
            const __m512i v1 = _mm512_loadu_si512((const void*)(c1 + i));
            const __m512i v2 = _mm512_loadu_si512((const void*)(c2 + i));
            const __m512i v3 = _mm512_loadu_si512((const void*)(c3 + i));

            acc0 = _mm512_dpbf16_ps(acc0, bq, (__m512bh)v0);
            acc1 = _mm512_dpbf16_ps(acc1, bq, (__m512bh)v1);
            acc2 = _mm512_dpbf16_ps(acc2, bq, (__m512bh)v2);
            acc3 = _mm512_dpbf16_ps(acc3, bq, (__m512bh)v3);
        }

        float r0 = _mm512_reduce_add_ps(acc0);
        float r1 = _mm512_reduce_add_ps(acc1);
        float r2 = _mm512_reduce_add_ps(acc2);
        float r3 = _mm512_reduce_add_ps(acc3);

        for (size_t i = d32; i < quant.d; i++) {
            const float qq = decode_bf16(qv[i]);
            r0 += qq * decode_bf16(c0[i]);
            r1 += qq * decode_bf16(c1[i]);
            r2 += qq * decode_bf16(c2[i]);
            r3 += qq * decode_bf16(c3[i]);
        }

        dis0 = r0;
        dis1 = r1;
        dis2 = r2;
        dis3 = r3;
    }

    void distances_batch(const idx_t* idx, float* dis, int n) override {
        if (n != 16) {
            for (int i = 0; i < n; i++) {
                const auto* c = (const uint16_t*)(codes + idx[i] * code_size);
                dis[i] = compute_code_ip_bf16(qbf16.data(), c);
            }
            return;
        }

        const size_t d32 = quant.d & ~size_t(31);
        if (d32 == 0) {
            for (int i = 0; i < 16; i++) {
                const auto* c = (const uint16_t*)(codes + idx[i] * code_size);
                dis[i] = compute_code_ip_bf16(qbf16.data(), c);
            }
            return;
        }

        static thread_local faiss::AlignedTable<uint16_t, 64> pack;
        pack.resize(16 * d32);
        for (int r = 0; r < 16; r++) {
            const auto* c = (const uint16_t*)(codes + idx[r] * code_size);
            std::memcpy(pack.data() + size_t(r) * d32, c, d32 * sizeof(uint16_t));
        }

        int rc = faiss::amx::ip_bf16_rows_1x16(
                pack.data(), qbf16.data(), d32, 16, dis);
        if (rc != 0) {
            for (int i = 0; i < 16; i++) {
                const auto* c = (const uint16_t*)(codes + idx[i] * code_size);
                dis[i] = compute_code_ip_bf16(qbf16.data(), c);
            }
            return;
        }

        for (int r = 0; r < 16; r++) {
            const auto* c = (const uint16_t*)(codes + idx[r] * code_size);
            for (size_t j = d32; j < quant.d; j++) {
                dis[r] += decode_bf16(qbf16[j]) * decode_bf16(c[j]);
            }
        }
    }

    float symmetric_dis(idx_t i, idx_t j) override {
        const auto* code1 = (const uint16_t*)(codes + i * code_size);
        const auto* code2 = (const uint16_t*)(codes + j * code_size);
        return compute_code_ip_bf16(code1, code2);
    }

    float query_to_code(const uint8_t* code) const final {
        const auto* c = (const uint16_t*)code;
        return compute_code_ip_bf16(qbf16.data(), c);
    }
};

#endif

template <class Quantizer, class Similarity>
struct DCTemplate<Quantizer, Similarity, SIMDLevel::NONE> : SQDistanceComputer {
    using Sim = Similarity;

    Quantizer quant;

    DCTemplate(size_t d, const std::vector<float>& trained)
            : quant(d, trained) {}

    float compute_distance(const float* x, const uint8_t* code) const {
        Similarity sim(x);
        sim.begin();
        for (size_t i = 0; i < quant.d; i++) {
            float xi = quant.reconstruct_component(code, i);
            sim.add_component(xi);
        }
        return sim.result();
    }

    float compute_code_distance(const uint8_t* code1, const uint8_t* code2)
            const {
        Similarity sim(nullptr);
        sim.begin();
        for (size_t i = 0; i < quant.d; i++) {
            float x1 = quant.reconstruct_component(code1, i);
            float x2 = quant.reconstruct_component(code2, i);
            sim.add_component_2(x1, x2);
        }
        return sim.result();
    }

    void set_query(const float* x) final {
        q = x;
    }

    float partial_dot_product(
            const idx_t i,
            const uint32_t offset,
            const uint32_t num_components) final override {
        if (i < 0) {
            FAISS_THROW_FMT(
                    "partial_dot_product called with negative id=%" PRId64,
                    (int64_t)i);
        }
        if (codes == nullptr) {
            FAISS_THROW_MSG("partial_dot_product called with null codes");
        }
        if (q == nullptr) {
            FAISS_THROW_MSG("partial_dot_product called with null query");
        }

        const uint32_t start = offset;
        const uint32_t end = std::min<uint32_t>(
                (uint32_t)quant.d,
                start + num_components);
        if (start >= end) {
            return 0.0f;
        }

        const size_t ii = (size_t)i;
        if (ii > (std::numeric_limits<size_t>::max() / code_size)) {
            FAISS_THROW_MSG("partial_dot_product id overflow");
        }
        const uint8_t* code = codes + ii * code_size;

        float res = 0.0f;
        for (uint32_t j = start; j < end; j++) {
            res += q[j] * quant.reconstruct_component(code, (size_t)j);
        }
        return res;
    }

    float symmetric_dis(idx_t i, idx_t j) override {
        return compute_code_distance(
                codes + i * code_size, codes + j * code_size);
    }

    float query_to_code(const uint8_t* code) const final {
        return compute_distance(q, code);
    }
};

#if defined(USE_AVX512_F16C)

template <class Quantizer, class Similarity>
struct DCTemplate<Quantizer, Similarity, SIMDLevel::AVX512>
        : SQDistanceComputer { // Update to handle 16 lanes
    using Sim = Similarity;

    Quantizer quant;

    DCTemplate(size_t d, const std::vector<float>& trained)
            : quant(d, trained) {}

    float compute_distance(const float* x, const uint8_t* code) const {
        Similarity sim(x);
        sim.begin_16();
        for (size_t i = 0; i < quant.d; i += 16) {
            simd16float32 xi = quant.reconstruct_16_components(code, i);
            sim.add_16_components(xi);
        }
        return sim.result_16();
    }

    float compute_code_distance(const uint8_t* code1, const uint8_t* code2)
            const {
        Similarity sim(nullptr);
        sim.begin_16();
        for (size_t i = 0; i < quant.d; i += 16) {
            simd16float32 x1 = quant.reconstruct_16_components(code1, i);
            simd16float32 x2 = quant.reconstruct_16_components(code2, i);
            sim.add_16_components_2(x1, x2);
        }
        return sim.result_16();
    }

    void set_query(const float* x) final {
        q = x;
    }

    float partial_dot_product(
            const idx_t i,
            const uint32_t offset,
            const uint32_t num_components) final override {
        if (i < 0) {
            FAISS_THROW_FMT(
                    "partial_dot_product called with negative id=%" PRId64,
                    (int64_t)i);
        }
        if (codes == nullptr) {
            FAISS_THROW_MSG("partial_dot_product called with null codes");
        }
        if (q == nullptr) {
            FAISS_THROW_MSG("partial_dot_product called with null query");
        }

        const uint32_t start = offset;
        const uint32_t end = std::min<uint32_t>(
                (uint32_t)quant.d,
                start + num_components);
        if (start >= end) {
            return 0.0f;
        }

        const size_t ii = (size_t)i;
        if (ii > (std::numeric_limits<size_t>::max() / code_size)) {
            FAISS_THROW_MSG("partial_dot_product id overflow");
        }
        const uint8_t* code = codes + ii * code_size;

        float res = 0.0f;
        uint32_t j = start;

        // Scalar head until 16-aligned.
        for (; j < end && (j & 15u) != 0; j++) {
            res += q[j] * quant.reconstruct_component(code, (size_t)j);
        }

        const uint32_t end16 = end & ~15u;
        for (; j < end16; j += 16) {
            const __m512 vq = _mm512_loadu_ps(q + j);
            const simd16float32 vx = quant.reconstruct_16_components(
                    code,
                    static_cast<int>(j));
            const __m512 prod = _mm512_mul_ps(vq, vx.f);
            res += _mm512_reduce_add_ps(prod);
        }

        // Scalar tail.
        for (; j < end; j++) {
            res += q[j] * quant.reconstruct_component(code, (size_t)j);
        }

        return res;
    }

    float symmetric_dis(idx_t i, idx_t j) override {
        return compute_code_distance(
                codes + i * code_size, codes + j * code_size);
    }

    float query_to_code(const uint8_t* code) const final {
        return compute_distance(q, code);
    }
};

#endif

#if defined(USE_F16C)

template <class Quantizer, class Similarity>
struct DCTemplate<Quantizer, Similarity, SIMDLevel::AVX2> : SQDistanceComputer {
    using Sim = Similarity;

    Quantizer quant;

    DCTemplate(size_t d, const std::vector<float>& trained)
            : quant(d, trained) {}

    float compute_distance(const float* x, const uint8_t* code) const {
        Similarity sim(x);
        sim.begin_8();
        for (size_t i = 0; i < quant.d; i += 8) {
            simd8float32 xi =
                    quant.reconstruct_8_components(code, static_cast<int>(i));
            sim.add_8_components(xi);
        }
        return sim.result_8();
    }

    float compute_code_distance(const uint8_t* code1, const uint8_t* code2)
            const {
        Similarity sim(nullptr);
        sim.begin_8();
        for (size_t i = 0; i < quant.d; i += 8) {
            simd8float32 x1 =
                    quant.reconstruct_8_components(code1, static_cast<int>(i));
            simd8float32 x2 =
                    quant.reconstruct_8_components(code2, static_cast<int>(i));
            sim.add_8_components_2(x1, x2);
        }
        return sim.result_8();
    }

    void set_query(const float* x) final {
        q = x;
    }

    FAISS_ALWAYS_INLINE static float reduce_add_ps(__m256 v) {
        const __m128 lo = _mm256_castps256_ps128(v);
        const __m128 hi = _mm256_extractf128_ps(v, 1);
        __m128 sum = _mm_add_ps(lo, hi);
        sum = _mm_hadd_ps(sum, sum);
        sum = _mm_hadd_ps(sum, sum);
        return _mm_cvtss_f32(sum);
    }

    float partial_dot_product(
            const idx_t i,
            const uint32_t offset,
            const uint32_t num_components) final override {
        if (i < 0) {
            FAISS_THROW_FMT(
                    "partial_dot_product called with negative id=%" PRId64,
                    (int64_t)i);
        }
        if (codes == nullptr) {
            FAISS_THROW_MSG("partial_dot_product called with null codes");
        }
        if (q == nullptr) {
            FAISS_THROW_MSG("partial_dot_product called with null query");
        }

        const uint32_t start = offset;
        const uint32_t end = std::min<uint32_t>(
                (uint32_t)quant.d,
                start + num_components);
        if (start >= end) {
            return 0.0f;
        }

        const size_t ii = (size_t)i;
        if (ii > (std::numeric_limits<size_t>::max() / code_size)) {
            FAISS_THROW_MSG("partial_dot_product id overflow");
        }
        const uint8_t* code = codes + ii * code_size;

        float res = 0.0f;
        uint32_t j = start;

        // Scalar head until 8-aligned.
        for (; j < end && (j & 7u) != 0; j++) {
            res += q[j] * quant.reconstruct_component(code, (size_t)j);
        }

        const uint32_t end8 = end & ~7u;
        for (; j < end8; j += 8) {
            const __m256 vq = _mm256_loadu_ps(q + j);
            const simd8float32 vx = quant.reconstruct_8_components(
                    code,
                    static_cast<int>(j));
            const __m256 prod = _mm256_mul_ps(vq, vx.f);
            res += reduce_add_ps(prod);
        }

        // Scalar tail.
        for (; j < end; j++) {
            res += q[j] * quant.reconstruct_component(code, (size_t)j);
        }

        return res;
    }

    float symmetric_dis(idx_t i, idx_t j) override {
        return compute_code_distance(
                codes + i * code_size, codes + j * code_size);
    }

    float query_to_code(const uint8_t* code) const final {
        return compute_distance(q, code);
    }
};

#endif

#ifdef USE_NEON

template <class Quantizer, class Similarity>
struct DCTemplate<Quantizer, Similarity, SIMDLevel::ARM_NEON>
        : SQDistanceComputer {
    using Sim = Similarity;

    Quantizer quant;

    DCTemplate(size_t d, const std::vector<float>& trained)
            : quant(d, trained) {}

    float compute_distance(const float* x, const uint8_t* code) const {
        Similarity sim(x);
        sim.begin_8();
        for (size_t i = 0; i < quant.d; i += 8) {
            simd8float32 xi =
                    quant.reconstruct_8_components(code, static_cast<int>(i));
            sim.add_8_components(xi);
        }
        return sim.result_8();
    }

    float compute_code_distance(const uint8_t* code1, const uint8_t* code2)
            const {
        Similarity sim(nullptr);
        sim.begin_8();
        for (size_t i = 0; i < quant.d; i += 8) {
            simd8float32 x1 =
                    quant.reconstruct_8_components(code1, static_cast<int>(i));
            simd8float32 x2 =
                    quant.reconstruct_8_components(code2, static_cast<int>(i));
            sim.add_8_components_2(x1, x2);
        }
        return sim.result_8();
    }

    void set_query(const float* x) final {
        q = x;
    }

    float partial_dot_product(
            const idx_t i,
            const uint32_t offset,
            const uint32_t num_components) final override {
        if (i < 0) {
            FAISS_THROW_FMT(
                    "partial_dot_product called with negative id=%" PRId64,
                    (int64_t)i);
        }
        if (codes == nullptr) {
            FAISS_THROW_MSG("partial_dot_product called with null codes");
        }
        if (q == nullptr) {
            FAISS_THROW_MSG("partial_dot_product called with null query");
        }

        const uint32_t start = offset;
        const uint32_t end = std::min<uint32_t>(
                (uint32_t)quant.d,
                start + num_components);
        if (start >= end) {
            return 0.0f;
        }

        const size_t ii = (size_t)i;
        if (ii > (std::numeric_limits<size_t>::max() / code_size)) {
            FAISS_THROW_MSG("partial_dot_product id overflow");
        }
        const uint8_t* code = codes + ii * code_size;

        float res = 0.0f;
        for (uint32_t j = start; j < end; j++) {
            res += q[j] * quant.reconstruct_component(code, (size_t)j);
        }
        return res;
    }

    float symmetric_dis(idx_t i, idx_t j) override {
        return compute_code_distance(
                codes + i * code_size, codes + j * code_size);
    }

    float query_to_code(const uint8_t* code) const final {
        return compute_distance(q, code);
    }
};

#endif

/*******************************************************************
 * DistanceComputerByte: computes distances in the integer domain
 *******************************************************************/

template <class Similarity, SIMDLevel SL>
struct DistanceComputerByte : SQDistanceComputer {};

template <class Similarity>
struct DistanceComputerByte<Similarity, SIMDLevel::NONE> : SQDistanceComputer {
    using Sim = Similarity;

    int d;
    std::vector<uint8_t> tmp;

    DistanceComputerByte(int d, const std::vector<float>&) : d(d), tmp(d) {}

    int compute_code_distance(const uint8_t* code1, const uint8_t* code2)
            const {
        int accu = 0;
        for (int i = 0; i < d; i++) {
            if (Sim::metric_type == METRIC_INNER_PRODUCT) {
                accu += int(code1[i]) * code2[i];
            } else {
                int diff = int(code1[i]) - code2[i];
                accu += diff * diff;
            }
        }
        return accu;
    }

    void set_query(const float* x) final {
        for (int i = 0; i < d; i++) {
            tmp[i] = int(x[i]);
        }
    }

    int compute_distance(const float* x, const uint8_t* code) {
        set_query(x);
        return compute_code_distance(tmp.data(), code);
    }

    float symmetric_dis(idx_t i, idx_t j) override {
        return compute_code_distance(
                codes + i * code_size, codes + j * code_size);
    }

    float query_to_code(const uint8_t* code) const final {
        return compute_code_distance(tmp.data(), code);
    }
};

#if defined(__AVX512F__)

template <class Similarity>
struct DistanceComputerByte<Similarity, SIMDLevel::AVX512>
        : SQDistanceComputer {
    using Sim = Similarity;

    int d;
    std::vector<uint8_t> tmp;

    DistanceComputerByte(int d, const std::vector<float>&) : d(d), tmp(d) {}

    int compute_code_distance(const uint8_t* code1, const uint8_t* code2)
            const {
        __m512i accu = _mm512_setzero_si512();
        for (int i = 0; i < d; i += 32) { // Process 32 bytes at a time
            __m512i c1 = _mm512_cvtepu8_epi16(
                    _mm256_loadu_si256((__m256i*)(code1 + i)));
            __m512i c2 = _mm512_cvtepu8_epi16(
                    _mm256_loadu_si256((__m256i*)(code2 + i)));
            __m512i prod32;
            if (Sim::metric_type == METRIC_INNER_PRODUCT) {
                prod32 = _mm512_madd_epi16(c1, c2);
            } else {
                __m512i diff = _mm512_sub_epi16(c1, c2);
                prod32 = _mm512_madd_epi16(diff, diff);
            }
            accu = _mm512_add_epi32(accu, prod32);
        }
        // Horizontally add elements of accu
        return _mm512_reduce_add_epi32(accu);
    }

    void set_query(const float* x) final {
        for (int i = 0; i < d; i++) {
            tmp[i] = int(x[i]);
        }
    }

    int compute_distance(const float* x, const uint8_t* code) {
        set_query(x);
        return compute_code_distance(tmp.data(), code);
    }

    float symmetric_dis(idx_t i, idx_t j) override {
        return compute_code_distance(
                codes + i * code_size, codes + j * code_size);
    }

    float query_to_code(const uint8_t* code) const final {
        return compute_code_distance(tmp.data(), code);
    }
};

#elif defined(__AVX2__)

template <class Similarity>
struct DistanceComputerByte<Similarity, SIMDLevel::AVX2> : SQDistanceComputer {
    using Sim = Similarity;

    int d;
    std::vector<uint8_t> tmp;

    DistanceComputerByte(int d, const std::vector<float>&) : d(d), tmp(d) {}

    int compute_code_distance(const uint8_t* code1, const uint8_t* code2)
            const {
        // __m256i accu = _mm256_setzero_ps ();
        __m256i accu = _mm256_setzero_si256();
        for (int i = 0; i < d; i += 16) {
            // load 16 bytes, convert to 16 uint16_t
            __m256i c1 = _mm256_cvtepu8_epi16(
                    _mm_loadu_si128((__m128i*)(code1 + i)));
            __m256i c2 = _mm256_cvtepu8_epi16(
                    _mm_loadu_si128((__m128i*)(code2 + i)));
            __m256i prod32;
            if (Sim::metric_type == METRIC_INNER_PRODUCT) {
                prod32 = _mm256_madd_epi16(c1, c2);
            } else {
                __m256i diff = _mm256_sub_epi16(c1, c2);
                prod32 = _mm256_madd_epi16(diff, diff);
            }
            accu = _mm256_add_epi32(accu, prod32);
        }
        __m128i sum = _mm256_extractf128_si256(accu, 0);
        sum = _mm_add_epi32(sum, _mm256_extractf128_si256(accu, 1));
        sum = _mm_hadd_epi32(sum, sum);
        sum = _mm_hadd_epi32(sum, sum);
        return _mm_cvtsi128_si32(sum);
    }

    void set_query(const float* x) final {
        /*
        for (int i = 0; i < d; i += 8) {
            __m256 xi = _mm256_loadu_ps (x + i);
            __m256i ci = _mm256_cvtps_epi32(xi);
        */
        for (int i = 0; i < d; i++) {
            tmp[i] = int(x[i]);
        }
    }

    int compute_distance(const float* x, const uint8_t* code) {
        set_query(x);
        return compute_code_distance(tmp.data(), code);
    }

    float symmetric_dis(idx_t i, idx_t j) override {
        return compute_code_distance(
                codes + i * code_size, codes + j * code_size);
    }

    float query_to_code(const uint8_t* code) const final {
        return compute_code_distance(tmp.data(), code);
    }
};

#endif

#ifdef USE_NEON

template <class Similarity>
struct DistanceComputerByte<Similarity, SIMDLevel::ARM_NEON>
        : SQDistanceComputer {
    using Sim = Similarity;

    int d;
    std::vector<uint8_t> tmp;

    DistanceComputerByte(int d, const std::vector<float>&) : d(d), tmp(d) {}

    int compute_code_distance(const uint8_t* code1, const uint8_t* code2)
            const {
        int accu = 0;
        for (int i = 0; i < d; i++) {
            if (Sim::metric_type == METRIC_INNER_PRODUCT) {
                accu += int(code1[i]) * code2[i];
            } else {
                int diff = int(code1[i]) - code2[i];
                accu += diff * diff;
            }
        }
        return accu;
    }

    void set_query(const float* x) final {
        for (int i = 0; i < d; i++) {
            tmp[i] = int(x[i]);
        }
    }

    int compute_distance(const float* x, const uint8_t* code) {
        set_query(x);
        return compute_code_distance(tmp.data(), code);
    }

    float symmetric_dis(idx_t i, idx_t j) override {
        return compute_code_distance(
                codes + i * code_size, codes + j * code_size);
    }

    float query_to_code(const uint8_t* code) const final {
        return compute_code_distance(tmp.data(), code);
    }
};

#endif

} // namespace scalar_quantizer
} // namespace faiss
