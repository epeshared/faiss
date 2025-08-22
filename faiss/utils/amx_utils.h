#pragma once
#if defined(ENABLE_AMX) && defined(__AVX512F__)
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <immintrin.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <iostream>
#include <cstring>

namespace faiss {

#define XFEATURE_XTILECFG           17
#define XFEATURE_XTILEDATA          18
#define XFEATURE_MASK_XTILECFG      (1 << XFEATURE_XTILECFG)
#define XFEATURE_MASK_XTILEDATA     (1 << XFEATURE_XTILEDATA)
#define XFEATURE_MASK_XTILE         (XFEATURE_MASK_XTILECFG | XFEATURE_MASK_XTILEDATA)
#define ARCH_GET_XCOMP_PERM         0x1022
#define ARCH_REQ_XCOMP_PERM         0x1023

static int enable_amx() {
    unsigned long bitmask = 0;
    long status = syscall(SYS_arch_prctl, ARCH_GET_XCOMP_PERM, &bitmask);
    if (0 != status) {
        std::cout << "SYS_arch_prctl(READ) error" << std::endl;
        return -1;
    }
    if (bitmask & XFEATURE_MASK_XTILEDATA) {
        return -1;
    }
    status = syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA);
    if (0 != status) {
        return -1;
    }
    status = syscall(SYS_arch_prctl, ARCH_GET_XCOMP_PERM, &bitmask);
    if (0 != status || !(bitmask & XFEATURE_MASK_XTILEDATA)) {
        return -1;
    }
    return 0;
} // enable_amx

__attribute__((constructor)) static void library_load() {
    // this functionn will be automatically called when the library is loaded
    printf("###### Library loaded, enable amx\n");
    enable_amx();
    // _tile_loadconfig((void *)&cfg);
}


// 将float转换为bfloat16 (uint16_t存储)
uint16_t float_to_bf16(float f) {
    uint32_t val;
    std::memcpy(&val, &f, sizeof(float));
    return static_cast<uint16_t>(val >> 16);
}

// 将bfloat16转换为float
float bf16_to_float(uint16_t bf16) {
    uint32_t val = static_cast<uint32_t>(bf16) << 16;
    float f;
    std::memcpy(&f, &val, sizeof(float));
    return f;
}

static inline float ip_bf16_avx512(const uint16_t* x,
                                   const uint16_t* y,
                                   size_t dim) {
    __m512 vr = _mm512_setzero_ps();
    size_t i = 0;
    for (; i + 32 <= dim; i += 32) {
        __m512i vx = _mm512_loadu_si512(x + i);
        __m512i vy = _mm512_loadu_si512(y + i);
        vr = _mm512_dpbf16_ps(vr,
             reinterpret_cast<__m512bh&>(vx),
             reinterpret_cast<__m512bh&>(vy));
    }
    float tmp[16]; _mm512_storeu_ps(tmp, vr);
    float sum = 0; for (int k = 0; k < 16; ++k) sum += tmp[k];
    for (; i < dim; ++i) sum += bf16_to_float(x[i]) * bf16_to_float(y[i]);
    return sum;
}

} // faiss namespace
#endif // ENABLE_AMX && __AVX512F__
