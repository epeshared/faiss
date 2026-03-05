#pragma once

#include <cstddef>
#include <cstdint>

namespace faiss {
namespace amx {

// Optional stats for verifying that AMX tile kernels are actually executed.
// Enabled only when FAISS_AMX_BF16_STATS is set in the environment.
// These functions have C linkage to make it easy to query via ctypes/dlsym.
extern "C" {
uint64_t faiss_amx_ip_bf16_rows_1x16_calls();
void faiss_amx_ip_bf16_rows_1x16_reset();
}

// Enable AMX tile data for the calling thread (Linux arch_prctl).
// Returns true on success. Safe to call multiple times.
bool enable_amx_for_this_thread();

// Compute inner products between A rows (BF16) and q (BF16): out[r] = dot(A[r], q).
// - A is row-major with row stride = d elements (BF16).
// - rows must be in [1, 16].
// - d is number of BF16 elements and must be a multiple of 32.
// Returns 0 on success, non-zero on failure.
int ip_bf16_rows_1x16(
        const uint16_t* A,
        const uint16_t* q,
        size_t d,
        int rows,
        float* out);

} // namespace amx
} // namespace faiss
