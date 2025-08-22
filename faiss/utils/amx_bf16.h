#pragma once
#if defined(ENABLE_AMX) && defined(__AVX512F__)

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <immintrin.h>
#include <sys/syscall.h>
#include <unistd.h>

#include "amx_utils.h"

namespace faiss {

struct alignas(64) TileConfig {
    uint8_t  paletteId = 1;
    uint8_t  startRow  = 0;
    uint8_t  reserved[14] = {0};
    uint16_t colsb[16]   = {0};
    uint8_t  rows[16]    = {0};

    TileConfig(uint32_t DIM, uint32_t batchB) {
        // Layout: 3 tiles for 96 dims + interleaved query tiles
        const uint16_t tileBytesA = DIM * 2;
        const uint16_t tileBytesB = batchB * 2 * 2;
        // define tiles: M0/M1/M2 for library, Q1/Q4/Q6 for query
        colsb[0] = tileBytesA; rows[0] = 16;
        colsb[1] = tileBytesB; rows[1] = DIM / 2;
        colsb[2] = tileBytesA; rows[3] = 16;
        colsb[3] = tileBytesB; rows[2] = DIM / 2;
        colsb[4] = tileBytesA; rows[5] = 16;
        colsb[5] = tileBytesB; rows[4] = DIM / 2;
                                        
        colsb[6] = tileBytesB; rows[6] = DIM / 2;
        colsb[7] = tileBytesB; rows[7] = DIM / 2;
        // remaining tiles unused
    }
};

static_assert(sizeof(TileConfig) == 64, "TileConfig must be 64 bytes");

static inline void amx_ip_bf16_matrix(
        char** lib, const char* query,
        size_t dims, size_t batchA, size_t batchB,
        float* out, bool need_reload_query) {
    constexpr int DIM = 32;
    constexpr int BLOCK = 96;
    size_t full_blocks = dims / BLOCK;
    size_t tail = dims % BLOCK;

    thread_local TileConfig cfg(DIM, batchB);
    thread_local bool cfg_loaded = false;

    if (!cfg_loaded) {
        _tile_loadconfig(&cfg);
        cfg_loaded = true;
    }

     _tile_zero(6);
     _tile_zero(7);
    
    // printf("[amx] Zeroed tile 2\n");
    alignas(64) static uint8_t bufA[3][1024];
    // printf("[amx] Buffer A allocated: %zu bytes\n", sizeof(bufA));

    static thread_local char* last_lib_ptr = nullptr;

    int strideA = (DIM * sizeof(uint16_t)) / 8;
    // Process full 96-dim blocks
    for (size_t b = 0; b < full_blocks; ++b) {
        // printf("[amx] Processing block %zu/%zu\n", b, full_blocks);
        size_t off = b * BLOCK * sizeof(uint16_t);
        // if (b + 1 < full_blocks) {
        //     _mm_prefetch(lib[0] + off + BLOCK * sizeof(uint16_t), _MM_HINT_T0);
        //     // printf("[AMX] Prefetched next block data\n");
        // }        
        if (need_reload_query) {
          _tile_loadd(1, query + off, 4);
          _tile_loadd(3, query + off + 64, 4);
          _tile_loadd(5, query + off + 128, 4);
        }

        // printf("[amx] Block offset: %zu\n", off);
        for (size_t i = 0; i < batchA; ++i) {
            uint8_t* baseA = reinterpret_cast<uint8_t*>(lib[i]) + off;
            for (int t = 0; t < 3; ++t) {
                _mm512_store_si512(
                    reinterpret_cast<__m512i*>(bufA[t] + i * DIM * 2),
                    _mm512_loadu_si512(baseA + t * DIM * 2)
                );                 
            }
            // _mm_prefetch(bufA[0], _MM_HINT_T0);         
        }
        _tile_loadd(0, bufA[0], 64);
        _tile_loadd(2, bufA[1], 64);
        _tile_loadd(4, bufA[2], 64);     

        _tile_dpbf16ps(6, 0, 1);
        _tile_dpbf16ps(6, 2, 3);        
        _tile_dpbf16ps(6, 4, 5);
        // printf("[amx] Block %zu done\n", b);
    }

    // Process tail dims >= DIM
    size_t idx = full_blocks * BLOCK;
    for (size_t t = DIM; t <= tail; t += DIM) {
        // printf("[amx] Processing tail segment at idx=%zu\n", idx);
        for (size_t i = 0; i < batchA; ++i) {
            _mm512_store_si512(
                reinterpret_cast<__m512i*>(bufA[0] + i * DIM * 2),
                _mm512_loadu_si512(
                    reinterpret_cast<const uint8_t*>(lib[i]) + idx * sizeof(uint16_t)
                )
            );
        }
        _tile_loadd(0, bufA[0], 64);
        _tile_loadd(3, query + idx * sizeof(uint16_t), 4);
        _tile_dpbf16ps(6, 0, 3);
        idx += DIM;
        // printf("[amx] Tail segment done, new idx=%zu\n", idx);
    }

    // printf("[amx] Storing results to output\n");
    _tile_stored(6, out, batchB * sizeof(float));
    // printf("[amx] AMX inner product complete\n");
}

// Single vector BF16 inner product using AVX-512
// static inline float ip_bf16_avx512(const uint16_t* x,
//                                    const uint16_t* y,
//                                    size_t dim) {
//     __m512 vr = _mm512_setzero_ps();
//     size_t i = 0;
//     for (; i + 32 <= dim; i += 32) {
//         __m512i vx = _mm512_loadu_si512(x + i);
//         __m512i vy = _mm512_loadu_si512(y + i);
//         vr = _mm512_dpbf16_ps(vr,
//              reinterpret_cast<__m512bh&>(vx),
//              reinterpret_cast<__m512bh&>(vy));
//     }
//     float tmp[16]; _mm512_storeu_ps(tmp, vr);
//     float sum = 0;
//     for (int k = 0; k < 16; ++k) sum += tmp[k];
//     for (; i < dim; ++i) sum += float(x[i]) * float(y[i]);
//     return sum;
// }

static inline void ip_bf16_batch(
    void** library_vector_ptrs,
    const void*  query_data_ptr,
    size_t       vector_dim,
    size_t       num_vectors,
    size_t       num_queries,
    float*       output_scores)
{
    // reinterpret pointers
    char**       vector_ptrs  = reinterpret_cast<char**>(library_vector_ptrs);
    const char*  query_bytes  = reinterpret_cast<const char*>(query_data_ptr);

    // block sizes
    constexpr size_t VECTOR_BLOCK = 16;
    constexpr size_t QUERY_BLOCK  = 16;

    // number of full blocks and remainder sizes
    size_t num_vector_blocks       = (num_vectors + VECTOR_BLOCK - 1) / VECTOR_BLOCK;
    size_t num_query_blocks        = (num_queries + QUERY_BLOCK - 1) / QUERY_BLOCK;
    size_t last_vector_block_size  = num_vectors % VECTOR_BLOCK ? (num_vectors % VECTOR_BLOCK) : VECTOR_BLOCK;
    size_t last_query_block_size   = num_queries % QUERY_BLOCK  ? (num_queries  % QUERY_BLOCK)  : QUERY_BLOCK;

    float* out_ptr = output_scores;

    // outer loop over query blocks
    for (size_t q_block_idx = 0; q_block_idx < num_query_blocks; ++q_block_idx) {
        size_t query_block_size = (q_block_idx == num_query_blocks - 1)
                                  ? last_query_block_size
                                  : QUERY_BLOCK;
        const char* query_block_ptr = query_bytes
                                    + q_block_idx * QUERY_BLOCK * vector_dim * sizeof(uint16_t);

        // inner loop over vector blocks
        for (size_t v_block_idx = 0; v_block_idx < num_vector_blocks; ++v_block_idx) {
            size_t vector_block_size = (v_block_idx == num_vector_blocks - 1)
                                       ? last_vector_block_size
                                       : VECTOR_BLOCK;
            char** vector_block_ptr = vector_ptrs + v_block_idx * VECTOR_BLOCK;

            // reload_query only on the first vector block of each query block
            bool reload_query = (v_block_idx == 0);

            amx_ip_bf16_matrix(
                vector_block_ptr,
                query_block_ptr,
                vector_dim,
                vector_block_size,
                query_block_size,
                out_ptr,
                reload_query
            );

            // advance output pointer by the number of scores written
            out_ptr += vector_block_size * query_block_size;
        }
    }
}



// Reference entry point
static float bf16_vec_inner_product_amx_ref(
     void** lib, const void* query, const void* dim_ptr,
     size_t b, size_t q, float* out) {
    size_t dim = *reinterpret_cast<const size_t*>(dim_ptr);
    size_t full32 = (dim / 32) * 32;
    ip_bf16_batch(lib, query, full32, b, q, out);
    const uint16_t* y_tail = reinterpret_cast<const uint16_t*>(query) + full32;
    size_t tail = dim - full32;
    if (tail) {
        for (size_t i = 0; i < b; ++i) {
            const uint16_t* x_tail =
                reinterpret_cast<const uint16_t*>(lib[i]) + full32;
            out[i] += ip_bf16_avx512(x_tail, y_tail, tail);
        }
    }
    return 0.0f;
}

} // namespace faiss
#endif // ENABLE_AMX && __AVX512F__
