#pragma once
#if defined(ENABLE_AMX) && defined(__AVX512F__)

#include <immintrin.h>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cassert>
#include <algorithm>
#include <stdio.h>

#include "amx_utils.h"

namespace faiss {

// ---------------- Tile Config ----------------
class TileConfigRow {
public:
    TileConfigRow() { std::memset(this, 0, sizeof(*this)); paletteId = 1; }
    // DIM 固定 32；rowsA 固定 16（预重排已零填充）；rowsC=当前 A 批有效行数；batchB=当前 B 批大小
    TileConfigRow(uint32_t DIM, uint32_t rowsA, uint32_t rowsC, uint32_t batchB) {
        std::memset(this, 0, sizeof(*this));
        paletteId = 1; startRow = 0;

        // A tiles：每行 32 bf16 = 64B，16 行
        colsb[0] = DIM * 2; rows[0] = (uint8_t)rowsA;
        colsb[3] = DIM * 2; rows[3] = (uint8_t)rowsA;
        colsb[5] = DIM * 2; rows[5] = (uint8_t)rowsA;

        // B tiles：行 = DIM/2 = 16；列跨度 = batchB * 4（保持你原代码的约定）
        uint16_t cstride = (uint16_t)(batchB * 4);
        colsb[1] = cstride; rows[1] = DIM / 2;
        colsb[4] = cstride; rows[4] = DIM / 2;
        colsb[6] = cstride; rows[6] = DIM / 2;

        // C (acc) tile：行 = 有效 A 行（rowsC），列跨度同 B
        colsb[2] = cstride; rows[2] = (uint8_t)rowsC;
    }
private:
    uint8_t  paletteId;
    uint8_t  startRow;
    uint8_t  reserved[14];
    uint16_t colsb[16];
    uint8_t  rows[16];
};

// 缓存最近一次的 config，按 (rowsC, batchB) 变化动态刷新
static inline void ensure_tile_config_packed(uint32_t rowsC, uint32_t batchB) {
    constexpr uint32_t DIM = 32, rowsA = 16;
    thread_local uint32_t last_rowsC = 0, last_b = 0;
    if (rowsC != last_rowsC || batchB != last_b) {
        TileConfigRow cfg(DIM, rowsA, rowsC, batchB);
        _tile_loadconfig((void*)&cfg);
        last_rowsC = rowsC; last_b = batchB;
    }
}

// ---------------- A 侧预重排（tile-friendly） ----------------
struct PackedA {
    uint8_t* buf = nullptr;       // 按 32 维分块后的 tile 序列
    size_t   groups = 0;          // 组数：ceil(nSize/16)
    size_t   kblocks = 0;         // K 方向块数：floor(dim/32)（<32 的尾巴不在这里）
    size_t   nSize = 0;           // 原始向量条数
    size_t   dim   = 0;           // 原始维度
    // 每组的字节跨度（= kblocks * 1024）
    size_t   bytes_per_group() const { return kblocks * 1024; }
};

// 将 void**（每条向量为连续 bf16[dim]）预重排：
// 目标布局：按 group(16 行) × kblock(32维) 顺序，存储 16×64B 的 tile；不足 16 行/不足 32 维零填充。
// AVX-512 优化版：一次 64B load/store 拷行；仅对最后一组不足行数写零；适度预取
static inline void prepack_A_bf16_tiles(void** pVect1v, size_t nSize, size_t dim, PackedA& out) {
    const size_t groups  = (nSize + 15) / 16;
    const size_t kblocks = (dim / 32);              // 只打包完整的 32-D 块
    const size_t bytes_per_group = kblocks * 1024;  // 16 行 × 64B × kblocks
    const size_t total_bytes = groups * bytes_per_group;

    // 分配 64B 对齐内存（无需整块 memset，后面只对需要的地方写零）
    void* p = nullptr;
    if (total_bytes) {
        int rc = posix_memalign(&p, 64, total_bytes);
        if (rc != 0 || !p) { perror("posix_memalign"); std::abort(); }
    }

    uint8_t* dst = (uint8_t*)p;
    // 常量零寄存器（用于最后一组补零）
    const __m512i zmm_zero = _mm512_setzero_si512();

    for (size_t g = 0; g < groups; ++g) {
        const size_t base_vec = g * 16;
        const size_t cur_rows = std::min<size_t>(16, nSize - base_vec); // 本组实际行数
        uint8_t* gptr = dst + g * bytes_per_group;

        for (size_t kb = 0; kb < kblocks; ++kb) {
            uint8_t* tile_ptr = gptr + kb * 1024;     // 本 tile 起始
            const size_t byte_off = kb * 64;          // 32×bf16 = 64B

            // 先拷真正存在的行（cur_rows）
            for (size_t r = 0; r < cur_rows; ++r) {
                const size_t vidx = base_vec + r;
                const uint8_t* row_src = (const uint8_t*)((const char*)pVect1v[vidx]) + byte_off;
                uint8_t* row_dst = tile_ptr + r * 64;

                // 预取下一块（对顺序 kb+1 有帮助；边界判断避免越界）
                if (kb + 1 < kblocks) {
                    _mm_prefetch((const char*)pVect1v[vidx] + (kb + 1) * 64, _MM_HINT_T0);
                }

                // 64B → 64B：unaligned load + aligned store
                __m512i v = _mm512_loadu_si512((const void*)row_src); // 源可能不对齐（vector/flat 情况）
                _mm512_store_si512((void*)row_dst, v);                // 目的地 64B 对齐
            }

            // 不足 16 行的补零（只在最后一组触发；等价于原先整块 memset 的语义要求）
            for (size_t r = cur_rows; r < 16; ++r) {
                uint8_t* row_dst = tile_ptr + r * 64;
                _mm512_store_si512((void*)row_dst, zmm_zero);
            }
        }
    }

    out.buf = dst; out.groups = groups; out.kblocks = kblocks; out.nSize = nSize; out.dim = dim;
}


// 释放预重排内存
static inline void free_packedA(PackedA& A) {
    if (A.buf) std::free(A.buf);
    A = PackedA{};
}

// // <32 的尾巴：一次掩码 dpbf16 收尾
static inline float dot_tail_bf16_mask(const uint16_t* x, const uint16_t* y, int rem) {
    if (rem <= 0) return 0.f;
    __mmask32 k = rem == 32 ? (__mmask32)0xFFFFFFFFu : ((__mmask32)1u<<rem) - 1u;
    __m512i vx_i = _mm512_maskz_loadu_epi16(k, x);
    __m512i vy_i = _mm512_maskz_loadu_epi16(k, y);
    __m512 acc = _mm512_setzero_ps();
    acc = _mm512_dpbf16_ps(acc, (__m512bh)vx_i, (__m512bh)vy_i);
    alignas(64) float tmp[16]; _mm512_store_ps(tmp, acc);
    float s=0.f; for (int i=0;i<16;++i) s+=tmp[i];
    return s;
}

// // 单个A组(<=16行) × 当前B批(<=16列) 主干 dim32 计算；写回为 q-major: out[q*batchA + i]
static inline void amx_block_group_qmajor(
    const PackedA& A, size_t group_idx,
    char* qBatch, uint32_t curA, uint32_t curB,
    float* out_for_q0, size_t batchA_total)
{
    if (!A.kblocks || !curA || !curB) return;
    ensure_tile_config_packed(curA, curB);
    _tile_zero(2);

    const size_t g_base = group_idx * A.bytes_per_group();
    const size_t full96 = (A.kblocks/3)*3;

    for (size_t kb=0; kb<full96; kb+=3) {
        const size_t b0=kb*64, b1=(kb+1)*64, b2=(kb+2)*64;
        _tile_loadd(1, qBatch + b0, 4);
        _tile_loadd(4, qBatch + b1, 4);
        _tile_loadd(6, qBatch + b2, 4);
        _tile_loadd(0, A.buf + g_base + (kb+0)*1024, 64);
        _tile_loadd(3, A.buf + g_base + (kb+1)*1024, 64);
        _tile_loadd(5, A.buf + g_base + (kb+2)*1024, 64);
        _tile_dpbf16ps(2,3,4);
        _tile_dpbf16ps(2,0,1);
        _tile_dpbf16ps(2,5,6);
    }
    const size_t rem = A.kblocks - full96;
    if (rem>=1){ _tile_loadd(1, qBatch + full96*64, 4);
                 _tile_loadd(0, A.buf + g_base + full96*1024, 64);
                 _tile_dpbf16ps(2,0,1); }
    if (rem>=2){ _tile_loadd(4, qBatch + (full96+1)*64, 4);
                 _tile_loadd(3, A.buf + g_base + (full96+1)*1024, 64);
                 _tile_dpbf16ps(2,3,4); }

    alignas(64) float tilebuf[16*16];
    _tile_stored(2, tilebuf, (int)(curB*4));      // row-major: curA×curB
    const size_t i_base = group_idx * 16;
    for (uint32_t r=0; r<curA; ++r)
        for (uint32_t c=0; c<curB; ++c)
            out_for_q0[c*batchA_total + (i_base + r)] = tilebuf[r*curB + c];
}

static inline void amx_row_scores_packed(uint16_t** lib_ptrs, uint16_t* query_data, size_t dim, size_t batchA, size_t batchB,
                                         const PackedA& A, float* out)
{
    std::fill(out, out + batchA * batchB, 0.f);
    const size_t dim32 = (dim/32)*32;
    const int tail = (int)(dim - dim32);

    const size_t Bblocks = (batchB + 15) / 16;
    for (size_t jb=0; jb<Bblocks; ++jb) {
        const size_t q0 = jb*16;
        const uint32_t curB = (uint32_t)std::min<size_t>(16, batchB - q0);
        char* qBatch = (char*)(query_data + q0*dim);
        float* out_q0 = out + q0*batchA;

        const size_t groups = A.groups;
        const uint32_t lastA = (A.nSize % 16 == 0) ? 16u : (uint32_t)(A.nSize % 16);
        for (size_t g=0; g<groups; ++g) {
            const uint32_t curA = (g == groups-1) ? lastA : 16u;
            amx_block_group_qmajor(A, g, qBatch, curA, curB, out_q0, batchA);
        }

        if (tail) { // 尾巴补加
            for (uint32_t c=0; c<curB; ++c) {
                const uint16_t* y_tail = query_data + (q0 + c)*dim + dim32;
                for (size_t i=0; i<batchA; ++i) {
                    const uint16_t* x_tail = reinterpret_cast<const uint16_t*>(lib_ptrs[i]) + dim32;
                    out_q0[c*batchA + i] += dot_tail_bf16_mask(x_tail, y_tail, tail);
                }
            }
        }
    }
    // _tile_release();
}

} // faiss namespace
#endif // ENABLE_AMX && __AVX512F__