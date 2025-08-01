#pragma once
#if defined(ENABLE_AMX) && defined(__AVX512F__)
#include <cstddef>
#include <cstdint>
#include <sys/time.h>
#include <sys/syscall.h> 
#include <unistd.h>
#include <iostream>
#include <immintrin.h>
#include <cassert>
#include <cstdio>
#include <string>
#include <cstring>
namespace faiss {

#define XFEATURE_XTILECFG           17
#define XFEATURE_XTILEDATA          18
#define XFEATURE_MASK_XTILECFG      (1 << XFEATURE_XTILECFG)
#define XFEATURE_MASK_XTILEDATA     (1 << XFEATURE_XTILEDATA)
#define XFEATURE_MASK_XTILE         (XFEATURE_MASK_XTILECFG | XFEATURE_MASK_XTILEDATA)
#define ARCH_GET_XCOMP_PERM         0x1022
#define ARCH_REQ_XCOMP_PERM         0x1023   

#include <immintrin.h>
#include <cstdint>
#include <cstddef>


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

class TileConfig {
  public: 
    TileConfig() {
      paletteId = 1; // must be 1
      startRow = 0; // must be 0
      for (int i = 0; i < 14; i++) {
        reserved[i] = 0; // reserved bytes
      }
      for (int i = 0; i < 16; i++) {
        colsb[i] = 0; // column sizes in bytes
        rows[i] = 0; // row sizes in rows
      }
    }
    TileConfig(uint32_t DIM, uint32_t batchSizeB) { 
        paletteId=1;
        startRow=0;

        colsb[0] = DIM*2; // 32 * sizeof(int16_t)
        rows[0] = 16;
        // matrix B need a layout rearragement

        colsb[1] = batchSizeB*2*2;
        rows[1] = DIM/2;

        colsb[2] = batchSizeB*2*2;
        rows[2] = DIM/2;
 
        colsb[3] = DIM*2;
        rows[3] = 16;

        colsb[4] = batchSizeB*2*2;
        rows[4] = DIM/2;

        colsb[5] = DIM*2;;
        rows[5] = 16;

        colsb[6] = batchSizeB*2*2;
        rows[6] = DIM/2;
    }

  private :
    // must be 1
    uint8_t paletteId;
    // must be 0
    uint8_t startRow;
    uint8_t reserved[14];
    // measured in bytes
    uint16_t colsb[16];
    // measured in rows
    uint8_t rows[16];
};

static float amx_inner_product_matrix_bf16( char **floatLibraryMatrix, char  *floatQueryMatrix, uint64_t dims,uint64_t batchSizeA,
                              uint64_t batchSizeB, float *results_ptr){
    constexpr int DIM=32;
    constexpr int blockDim = 96;
    int blockCount=((dims))/blockDim;
    size_t tailCount=dims%DIM;
    int tailBlock=dims%blockDim;
 
    thread_local TileConfig cfg(DIM, batchSizeB);
    thread_local bool init_mem=false;
 
    static unsigned char ma1Bf16[1024] __attribute__((aligned(64)));
    static unsigned char ma2Bf16[1024] __attribute__((aligned(64)));
    static unsigned char ma3Bf16[1024] __attribute__((aligned(64)));
 
    // static float results[16*16] __attribute__((aligned(64)))={0};
 
    if(!init_mem){
        init_mem = true;
        printf("init tile\n");
        _tile_loadconfig((void *)&cfg);
    }

    // printf("_tile_loadconfig finished\n");

    _tile_zero(2);
    for(size_t i=0;i<blockCount;i++){
      __m512i sa;
      size_t offset = i * blockDim *2;

      // printf("batchSizeA: %ld,  offset: %ld, DIM: %d\n", batchSizeA, offset, DIM);
      for(size_t j=0;j<batchSizeA;j++){
        size_t destOffset1 = j * DIM * 2;
        // printf("destOffset1: %ld, j: %d\n", destOffset1, j);
        // printf("floatLibraryMatrix[j]: %p, offset: %ld\n", floatLibraryMatrix[j], offset);
        // printf("floatLibraryMatrix[j] + offset: %d\n", *(floatLibraryMatrix[j] + offset));
 
        _mm512_store_si512(ma1Bf16 + destOffset1, _mm512_loadu_si512(floatLibraryMatrix[j] + offset));
        // printf("ma1Bf16[%d]: %d\n", j, ((uint16_t*)ma1Bf16)[0]);  
        _mm512_store_si512(ma2Bf16 + destOffset1, _mm512_loadu_si512(floatLibraryMatrix[j] + offset + 64));
        // printf("ma2Bf16[%d]: %d\n", j, ((uint16_t*)ma2Bf16)[0]);
        _mm512_store_si512(ma3Bf16 + destOffset1, _mm512_loadu_si512(floatLibraryMatrix[j] + offset + 128));
        // printf("ma3Bf16[%d]: %d\n", j, ((uint16_t*)ma3Bf16)[0]);
      }
      // printf("finished loading ma1Bf16, ma2Bf16, ma3Bf16, i=%d\n", i);
      // fflush(stdout);
 
      _tile_loadd(1,floatQueryMatrix + offset , 4);
      _tile_loadd(4,floatQueryMatrix + offset + 64 , 4);
      _tile_loadd(6,floatQueryMatrix + offset + 128, 4);
      _tile_loadd(0,ma1Bf16, 64);
      _tile_loadd(3,ma2Bf16, 64);
      _tile_loadd(5,ma3Bf16, 64);
      _tile_dpbf16ps(2,3,4);
      _tile_dpbf16ps(2,0,1);
      _tile_dpbf16ps(2,5,6);
    }

    // printf("finish amx ip\n");

    if(tailBlock >= DIM){
      for(size_t i=0;i<tailBlock/DIM;i++){
        __m512i sa;
        for(size_t j=0;j<batchSizeA;j++){  
          sa=_mm512_loadu_si512(floatLibraryMatrix[j]+blockCount*blockDim * 2 + i * DIM * 2 );
          _mm512_store_si512(ma1Bf16+j*DIM*2,sa);
        }
        _tile_loadd(0,ma1Bf16, 64);
        _tile_loadd(1,floatQueryMatrix + blockCount*blockDim*2 + i * DIM*2 , 4);
        _tile_dpbf16ps(2,0,1);
      }
    }
    _tile_stored(2, results_ptr, batchSizeB*2*2);

    // memcpy(results_ptr, results, batchSizeA * batchSizeB * sizeof(float));
    // printf("finishe memcpy\n");
 
    return 0;
}
 
static float InnerProductDistanceBf16AVX512(const void* a, const void* b, const void *qty_ptr) {
  float result[16] = {0.0f}; // Used to store intermediate results

  uint16_t *x = (uint16_t *)a;
  uint16_t *y = (uint16_t *)b;
  __m512 vr_f32 = _mm512_setzero_ps(); // Initialize the accumulation register to zero

  size_t dim = * (size_t*) qty_ptr;

  size_t i = 0;
  // Process 32 elements at a time (16 __bf16 elements are stored as 32 uint16_t in a __m512bh register)
  for (; i + 32 <= dim; i += 32) {
      // Load 32 uint16_t into a temporary __m512i register
      __m512i temp_x = _mm512_loadu_si512(x + i);
      __m512i temp_y = _mm512_loadu_si512(y + i);

      // Cast to __m512bh type
      __m512bh v1_f16 = reinterpret_cast<__m512bh&>(temp_x);
      __m512bh v2_f16 = reinterpret_cast<__m512bh&>(temp_y);

      // Compute the BF16 dot product and accumulate the result into vr_f32
      vr_f32 = _mm512_dpbf16_ps(vr_f32, v1_f16, v2_f16);
  }

  // Store the values from the vr_f32 register into the result array
  _mm512_storeu_ps(result, vr_f32);

  // Sum all elements of the result array to obtain the final dot product
  float dot_product = 0.0f;
  for (int j = 0; j < 16; j++) {
      dot_product += result[j];
  }

  // Handle remaining elements (less than 32)
  for (; i < dim; i++) {
      float x_val = (float)(uint16_t)(x[i]);
      float y_val = (float)(uint16_t)(y[i]);
      dot_product += x_val * y_val;
  }
  return dot_product;
}


static float InnerProductBatchExtAMXBF16(void **pVect1v, void *pVect2v, void *qty_ptr, 
    size_t nSize, size_t mSize, float * results_amx){
  unsigned int dims= *(unsigned int*)qty_ptr;
  char **floatLibraryMatrix = (char**) pVect1v;
  char *floatQueryMatrix = (char*) pVect2v;
 
  // The size of one amx tile  16x64 bytes, it can store 16 64-dimensional vectors. 
  // Each vector is stored as 32 bf16 elements, which are 2 bytes each.
  // So, each tile can store 16 * 32 * 2 = 1024 bytes.
  constexpr int batchSizeA = 16, batchSizeB = 16;  
  int batchCountA = (nSize - 1) / batchSizeA + 1;
  int batchCountB = (mSize - 1) / batchSizeB + 1;
 
  int lastBatchSizeA = (nSize % batchSizeA == 0) ? batchSizeA : nSize % batchSizeA;
  int lastBatchSizeB = (mSize % batchSizeB == 0) ? batchSizeB : mSize % batchSizeB;
 
  int offsetA = batchSizeA * dims * 2;
  int offsetB = batchSizeB * dims * 2;
 
  float *results_ptr = results_amx;

  // printf("batchCountA: %d\n", batchCountA);
  for (size_t i = 0; i < batchCountA; i++) {
      // printf("i: %d\n", i);
      size_t currentBatchSizeA = (i == batchCountA - 1) ? lastBatchSizeA : batchSizeA;
      char **currentLibraryMatrixPtr = floatLibraryMatrix + i * 16;

      for (size_t j = 0; j < batchCountB; j++) {
          size_t currentBatchSizeB = (j == batchCountB - 1) ? lastBatchSizeB : batchSizeB;
          char *currentQueryMatrixPtr = floatQueryMatrix + j * offsetB;
          // printf("currentBatchSizeA: %ld, currentBatchSizeB: %ld\n", currentBatchSizeA, currentBatchSizeB);
          amx_inner_product_matrix_bf16(currentLibraryMatrixPtr, currentQueryMatrixPtr, dims, currentBatchSizeA, currentBatchSizeB, results_ptr);
          // printf("results_ptr: %p\n", results_ptr);
          results_ptr += currentBatchSizeB * currentBatchSizeA;
      }
  }
 
  return 0;
}

static float bf16_vec_inner_product_amx_ref(void **p_bVect1v, void *p_qVect2v, void *dim_ptr, 
    size_t b_Size, size_t q_Size, float* results_amx) {
    size_t qty = *((size_t *) dim_ptr);
    size_t qty32 = qty >> 5 << 5;
    
    // printf("qty: %ld, qty32: %ld\n", qty, qty32);
    InnerProductBatchExtAMXBF16(p_bVect1v, p_qVect2v, &qty32,b_Size,q_Size,results_amx);
    // printf("InnerProductBatchExtAMXBF16 finished#######>\n");
 
    size_t qty_left = qty - qty32;

    // printf("qty_left:%ld\n", qty_left);
 
    uint16_t *pVect2 = (uint16_t *) p_qVect2v + qty32;
    if(qty_left>0){
        for(size_t i = 0; i < b_Size; i++) {            
            uint16_t *pVect1 = (uint16_t *) p_bVect1v[i] + qty32;
            results_amx[i] += InnerProductDistanceBf16AVX512(pVect1, pVect2, &qty_left);
        }
    }
    // printf("InnerProductDistanceBf16AVX512 finished\n");
    return 0;
}
} // namespace faiss
#endif // defined(ENABLE_AMX) && defined(__AVX512F__)