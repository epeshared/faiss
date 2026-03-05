#include <faiss/utils/amx_tile_bf16.h>

#include <cstring>
#include <atomic>
#include <cstdlib>

#if defined(__linux__)
#include <sys/syscall.h>
#include <unistd.h>
#endif

#if defined(__AMX_TILE__) && defined(__AMX_BF16__)
#include <immintrin.h>
#endif

namespace faiss {
namespace amx {

namespace {
std::atomic<uint64_t> g_ip_bf16_rows_1x16_calls{0};

inline bool amx_bf16_stats_enabled() {
    static const bool enabled = (std::getenv("FAISS_AMX_BF16_STATS") != nullptr);
    return enabled;
}
} // namespace

extern "C" uint64_t faiss_amx_ip_bf16_rows_1x16_calls() {
    return g_ip_bf16_rows_1x16_calls.load(std::memory_order_relaxed);
}

extern "C" void faiss_amx_ip_bf16_rows_1x16_reset() {
    g_ip_bf16_rows_1x16_calls.store(0, std::memory_order_relaxed);
}

#if defined(__linux__)
namespace {
constexpr int XFEATURE_XTILECFG = 17;
constexpr int XFEATURE_XTILEDATA = 18;
constexpr unsigned long XFEATURE_MASK_XTILECFG = (1UL << XFEATURE_XTILECFG);
constexpr unsigned long XFEATURE_MASK_XTILEDATA = (1UL << XFEATURE_XTILEDATA);
constexpr unsigned long XFEATURE_MASK_XTILE =
        (XFEATURE_MASK_XTILECFG | XFEATURE_MASK_XTILEDATA);
constexpr unsigned long ARCH_GET_XCOMP_PERM = 0x1022;
constexpr unsigned long ARCH_REQ_XCOMP_PERM = 0x1023;

static bool enable_amx_impl() {
    unsigned long bitmask = 0;
    long status = syscall(SYS_arch_prctl, ARCH_GET_XCOMP_PERM, &bitmask);
    if (status != 0) {
        return false;
    }
    if (bitmask & XFEATURE_MASK_XTILEDATA) {
        return true;
    }
    status = syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA);
    if (status != 0) {
        return false;
    }
    status = syscall(SYS_arch_prctl, ARCH_GET_XCOMP_PERM, &bitmask);
    if (status != 0) {
        return false;
    }
    return (bitmask & XFEATURE_MASK_XTILE) == XFEATURE_MASK_XTILE;
}
} // namespace
#endif

bool enable_amx_for_this_thread() {
#if !defined(__linux__)
    return false;
#else
    static thread_local int state = 0; // 0: unknown, 1: ok, -1: fail
    if (state == 0) {
        state = enable_amx_impl() ? 1 : -1;
    }
    return state == 1;
#endif
}

int ip_bf16_rows_1x16(
        const uint16_t* A,
        const uint16_t* q,
        size_t d,
        int rows,
        float* out) {
#if !defined(__AMX_TILE__) || !defined(__AMX_BF16__)
    (void)A;
    (void)q;
    (void)d;
    (void)rows;
    (void)out;
    return -1;
#else
    if (rows <= 0 || rows > 16) {
        return -2;
    }
    if ((d % 32) != 0) {
        return -4;
    }
    if (!enable_amx_for_this_thread()) {
        return -3;
    }

    if (amx_bf16_stats_enabled()) {
        g_ip_bf16_rows_1x16_calls.fetch_add(1, std::memory_order_relaxed);
    }

    constexpr int K = 32; // bf16 elements per block
    const size_t block_count = d / K;

    alignas(64) static thread_local unsigned char cfg[64];
    static thread_local int prev_rows = -1;

    const int A_rows = rows;
    const int N = 1;
    const int A_colsb = K * 2;      // 64 bytes per row
    const int B_colsb = N * 4;      // 4 bytes per row
    const int B_rows = K / 2;       // 16 rows
    const int C_colsb = N * 4;      // 4 bytes
    const int C_rows = A_rows;

    if (prev_rows != A_rows) {
        std::memset(cfg, 0, sizeof(cfg));
        cfg[0] = 1;

        // tile0: A
        cfg[16] = (unsigned char)A_colsb;
        cfg[48] = (unsigned char)A_rows;

        // tile1: B
        cfg[18] = (unsigned char)B_colsb;
        cfg[49] = (unsigned char)B_rows;

        // tile2: C
        cfg[20] = (unsigned char)C_colsb;
        cfg[50] = (unsigned char)C_rows;

        // tile3/4
        cfg[22] = (unsigned char)A_colsb;
        cfg[51] = (unsigned char)A_rows;
        cfg[24] = (unsigned char)B_colsb;
        cfg[52] = (unsigned char)B_rows;

        // tile5/6
        cfg[26] = (unsigned char)A_colsb;
        cfg[53] = (unsigned char)A_rows;
        cfg[28] = (unsigned char)B_colsb;
        cfg[54] = (unsigned char)B_rows;

        _tile_loadconfig((void*)cfg);
        prev_rows = A_rows;
    }

    _tile_zero(2);

    const int a_stride = (int)(d * 2); // bytes between A rows

    // Unroll by 3 blocks for throughput.
    size_t i = 0;
    for (i = 0; i < block_count / 3; i++) {
        const size_t b0 = (3 * i + 0) * K;
        const size_t b1 = (3 * i + 1) * K;
        const size_t b2 = (3 * i + 2) * K;

        _tile_loadd(0, (const void*)(A + b0), a_stride);
        _tile_loadd(1, (const void*)(q + b0), 4);

        _tile_loadd(3, (const void*)(A + b1), a_stride);
        _tile_loadd(4, (const void*)(q + b1), 4);

        _tile_loadd(5, (const void*)(A + b2), a_stride);
        _tile_loadd(6, (const void*)(q + b2), 4);

        _tile_dpbf16ps(2, 0, 1);
        _tile_dpbf16ps(2, 3, 4);
        _tile_dpbf16ps(2, 5, 6);
    }

    switch (block_count % 3) {
        case 0:
            break;
        case 1: {
            const size_t b0 = (3 * i + 0) * K;
            _tile_loadd(0, (const void*)(A + b0), a_stride);
            _tile_loadd(1, (const void*)(q + b0), 4);
            _tile_dpbf16ps(2, 0, 1);
            break;
        }
        case 2: {
            const size_t b0 = (3 * i + 0) * K;
            const size_t b1 = (3 * i + 1) * K;
            _tile_loadd(0, (const void*)(A + b0), a_stride);
            _tile_loadd(1, (const void*)(q + b0), 4);
            _tile_loadd(3, (const void*)(A + b1), a_stride);
            _tile_loadd(4, (const void*)(q + b1), 4);
            _tile_dpbf16ps(2, 0, 1);
            _tile_dpbf16ps(2, 3, 4);
            break;
        }
    }

    _tile_stored(2, (void*)out, 4);

    return 0;
#endif
}

} // namespace amx
} // namespace faiss
