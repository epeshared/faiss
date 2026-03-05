/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/impl/DistanceComputer.h>

namespace faiss {

NegativeDistanceComputer::NegativeDistanceComputer(DistanceComputer* basedis)
        : basedis(basedis) {}

void NegativeDistanceComputer::set_query(const float* x) {
    basedis->set_query(x);
}

float NegativeDistanceComputer::operator()(idx_t i) {
    return -(*basedis)(i);
}

void NegativeDistanceComputer::distances_batch(
        const idx_t* idx,
        float* dis,
        int n) {
    basedis->distances_batch(idx, dis, n);
    for (int i = 0; i < n; i++) {
        dis[i] = -dis[i];
    }
}

void NegativeDistanceComputer::distances_batch_4(
        const idx_t idx0,
        const idx_t idx1,
        const idx_t idx2,
        const idx_t idx3,
        float& dis0,
        float& dis1,
        float& dis2,
        float& dis3) {
    basedis->distances_batch_4(idx0, idx1, idx2, idx3, dis0, dis1, dis2, dis3);
    dis0 = -dis0;
    dis1 = -dis1;
    dis2 = -dis2;
    dis3 = -dis3;
}

float NegativeDistanceComputer::symmetric_dis(idx_t i, idx_t j) {
    return -basedis->symmetric_dis(i, j);
}

NegativeDistanceComputer::~NegativeDistanceComputer() {
    delete basedis;
}

} // namespace faiss
