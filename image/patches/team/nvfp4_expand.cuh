// KV-NVFP4 dla sparse-MLA SM120 (GLM-5.2) — ekspansja prologowa.
// Spec: LAYOUT-KV-NVFP4.md (team document, variant V1, 352 B/token).
//
// Storage per token (gmem, adresowanie plaskie idx*352):
//   [0:256)   512 x E2M1 (2/bajt, dim parzysty w dolnym nibblu)
//   [256:288) 32 x E4M3 skala blokowa (blok 16), dequant = e4m3 * 2^-6
//   [288:352) 64 x FP8 E4M3 rope, skala 1.0
//
// Kernel-side: IO laduje spakowane 288 B na KONIEC istniejacego slotu smem
// 528 B (offset 240), a rope 64 B na koniec slotu rope 128 B (offset 64).
// Math warpy przed QK rozpakowuja in-place do layoutu zgodnego z GLM_NSA:
//   [0:512)   E4M3 (wartosci przeskalowane do wspolnej skali tile-128)
//   [512:528) 4 x FP32 skala tile-128 (max ze skal blokowych w tile'u)
// dzieki czemu CALY istniejacy pipeline FP8-MMA dziala bez zmian.
// Dodatkowy szum requantu zmierzony na realnych latentach:
// rel-RMS 0.0949 -> 0.0967 (= tor bledow zwalidowany E2E przez fake-quant
// + zapis fp8_ds_mla, patrz KV-NVFP4-HANDOFF.md).
//
// Rope: dequant E4M3 -> BF16 in-place (dokladny, bez dodatkowego szumu).

#pragma once

#include <cuda_fp16.h>

// Globalny namespace — zgodnie z pozostalymi naglowkami common/ (q_rope itd.);
// decode_dsv3_2_kernel.cuh (namespace flashinfer::sparse_mla_sm120) tez widzi.

// E2M1 magnitudy {0,.5,1,1.5,2,3,4,6} jako bajty E4M3 (reprezentacja dokladna).
constexpr uint32_t NVFP4_LUT_A = 0x3C383000u;  // kody 0..3
constexpr uint32_t NVFP4_LUT_B = 0x4C484440u;  // kody 4..7
constexpr float NVFP4_GLOBAL_SCALE = 0.015625f;  // 2^-6, LAYOUT §2

// Offsety w gmem-blobie (352 B) i sloty smem.
constexpr int NVFP4_GMEM_SCALE_OFF = 256;
constexpr int NVFP4_GMEM_ROPE_OFF = 288;
constexpr int NVFP4_PACKED_BYTES = 288;      // nibble + skale (bulk 1)
constexpr int NVFP4_SMEM_LANDING_OFF = 240;  // 528 - 288
constexpr int NVFP4_ROPE_GMEM_BYTES = 64;    // bulk 2
constexpr int NVFP4_ROPE_LANDING_OFF = 64;   // 128 - 64

// 4 bajty = 8 nibbli E2M1 -> e4_lo (dimy parzyste), e4_hi (nieparzyste),
// kolejnosc bajtow = kolejnosc par. Zweryfikowane bit-exact w
// fp4_probe/bench/dequant_bench.cu (verify vs torch).
__device__ __forceinline__ void nvfp4_e2m1x8_to_e4m3(uint32_t w, uint32_t& e4_lo,
                                                     uint32_t& e4_hi) {
  const uint32_t lo = w & 0x0F0F0F0Fu;
  const uint32_t hi = (w >> 4) & 0x0F0F0F0Fu;
#pragma unroll
  for (int k = 0; k < 2; k++) {
    const uint32_t s = k ? hi : lo;
    const uint32_t mag = s & 0x07070707u;
    const uint32_t t = mag | (mag >> 4);
    const uint32_t sel = __byte_perm(t, 0u, 0x4420u);
    uint32_t e4 = __byte_perm(NVFP4_LUT_A, NVFP4_LUT_B, sel);
    e4 |= (s & 0x08080808u) << 4;
    (k ? e4_hi : e4_lo) = e4;
  }
}

__device__ __forceinline__ __half2 nvfp4_fp8x2_to_h2(uint32_t two_bytes) {
  uint32_t f16x2;
  asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(f16x2) : "h"((uint16_t)two_bytes));
  return *reinterpret_cast<__half2*>(&f16x2);
}

__device__ __forceinline__ float nvfp4_fp8_to_f32(uint8_t b) {
  return __half2float(__low2half(nvfp4_fp8x2_to_h2((uint32_t)b)));
}

// 4 bajty E4M3 (mag+znak) * ratio -> 4 bajty E4M3 (satfinite RN).
__device__ __forceinline__ uint32_t nvfp4_requant4(uint32_t e4, __half2 ratio2) {
  const __half2 a = __hmul2(nvfp4_fp8x2_to_h2(e4 & 0xFFFFu), ratio2);
  const __half2 b = __hmul2(nvfp4_fp8x2_to_h2(e4 >> 16), ratio2);
  uint16_t pa, pb;
  asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(pa) : "r"(*(const uint32_t*)&a));
  asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(pb) : "r"(*(const uint32_t*)&b));
  return (uint32_t)pa | ((uint32_t)pb << 16);
}

// 4 bajty E4M3 -> 2x u32 bf16x2 (dequant rope, dokladny).
__device__ __forceinline__ void nvfp4_fp8x4_to_bf16x4(uint32_t e4, uint32_t& b01,
                                                      uint32_t& b23) {
  const __half2 h01 = nvfp4_fp8x2_to_h2(e4 & 0xFFFFu);
  const __half2 h23 = nvfp4_fp8x2_to_h2(e4 >> 16);
  const float f0 = __low2float(h01), f1 = __high2float(h01);
  const float f2 = __low2float(h23), f3 = __high2float(h23);
  asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(b01) : "f"(f1), "f"(f0));
  asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(b23) : "f"(f3), "f"(f2));
}

// Prefetch rope B-operandow z GMEM dla prefillu: 64 x E4M3 zamiast bf16.
// Zwraca strukture zgodna z q_rope.cuh::KVRopePrefetch (szablonem, bo ten
// naglowek moze byc wlaczony przed q_rope.cuh). Mapowanie lane'ow identyczne
// z prefetch_kv_rope: b[ks][0] = elemy (16ks+2tid, +1), b[ks][1] = (+8).
template <typename KVRopePrefetchT, int N_ROPE_CHUNKS_T = 4>
__device__ __forceinline__ KVRopePrefetchT nvfp4_prefetch_kv_rope_t(
    const uint8_t* rope_packed, int lane) {
  const int tid = lane & 3;
  KVRopePrefetchT pf;
#pragma unroll
  for (int ks = 0; ks < N_ROPE_CHUNKS_T; ks++) {
    const int e0 = ks * 16 + tid * 2;
    const uint16_t p0 = *reinterpret_cast<const uint16_t*>(rope_packed + e0);
    const uint16_t p1 = *reinterpret_cast<const uint16_t*>(rope_packed + e0 + 8);
    const __half2 h0 = nvfp4_fp8x2_to_h2((uint32_t)p0);
    const __half2 h1 = nvfp4_fp8x2_to_h2((uint32_t)p1);
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;"
        : "=r"(pf.b[ks][0])
        : "f"(__high2float(h0)), "f"(__low2float(h0)));
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;"
        : "=r"(pf.b[ks][1])
        : "f"(__high2float(h1)), "f"(__low2float(h1)));
  }
  return pf;
}

// Ekspansja jednego kafla BI wpisow, in-place.
//   kv_smem:   BI slotow po smem_stride(=528) B; spakowane dane pod
//              [NVFP4_SMEM_LANDING_OFF, 528) kazdego slotu.
//   rope_smem: BI slotow po 128 B (bf16[64]); spakowane pod [64,128); moze
//              byc nullptr (prefill czyta rope z gmem).
// Wolane przez THREADS watkow (tid = 0..THREADS-1), BarSync = bariera
// obejmujaca DOKLADNIE te watki. Zewnetrzna widocznosc danych po bulk
// zapewnia mbarrier; po powrocie wymagana zewnetrzna bariera przed odczytem
// przez inne warpy niz ekspandujace (w praktyce: te same math warpy).
template <int THREADS, int BI_T, typename BarSync>
__device__ __forceinline__ void nvfp4_expand_tile(uint8_t* kv_smem, int smem_stride,
                                                  uint8_t* rope_smem, int tid,
                                                  BarSync bar) {
  // Praca: BI*16 segmentow latentu (segment = 32 dimy = 16 B packed
  // = 8 B out... 32 B out) + BI*16 grup rope (grupa = 4 elemy).
  constexpr int SEGS = BI_T * 16;
  constexpr int SEGS_PER_PHASE = SEGS / 2;
  constexpr int SEG_PER_THREAD = SEGS_PER_PHASE / THREADS;  // 2 dla BI=64,T=256
  static_assert(SEGS_PER_PHASE % THREADS == 0);

#pragma unroll
  for (int phase = 0; phase < 2; phase++) {
    // --- faza read/decode do rejestrow ---
    uint32_t out[SEG_PER_THREAD][8];
    float tile_sc[SEG_PER_THREAD];
    uint32_t rope_out[SEG_PER_THREAD][2];
#pragma unroll
    for (int i = 0; i < SEG_PER_THREAD; i++) {
      const int seg = phase * SEGS_PER_PHASE + i * THREADS + tid;
      const int cand = seg >> 4;
      const int s = seg & 15;  // ktory segment 32-dim w obrebie tokenu
      uint8_t* slot = kv_smem + (size_t)cand * smem_stride;
      const uint8_t* packed = slot + NVFP4_SMEM_LANDING_OFF;

      // skale blokowe segmentu (bloki 2s, 2s+1) i skala tile'a (max z 8)
      const uint8_t* scales = packed + 256;  // = slot + 496
      const int t128 = s >> 2;
      float tmax = 0.f;
#pragma unroll
      for (int b = 0; b < 8; b++)
        tmax = fmaxf(tmax, nvfp4_fp8_to_f32(scales[t128 * 8 + b]));
      const float sc0 = nvfp4_fp8_to_f32(scales[2 * s]);
      const float sc1 = nvfp4_fp8_to_f32(scales[2 * s + 1]);
      const float inv_t = (tmax > 0.f) ? (1.0f / tmax) : 0.f;
      const __half2 r0 = __half2half2(__float2half(sc0 * inv_t));
      const __half2 r1 = __half2half2(__float2half(sc1 * inv_t));
      tile_sc[i] = tmax * NVFP4_GLOBAL_SCALE;

      // 16 B packed nibbli -> 32 B e4m3 (przeskalowane do skali tile'a)
      const uint4 pk = *reinterpret_cast<const uint4*>(packed + s * 16);
      const uint32_t ws[4] = {pk.x, pk.y, pk.z, pk.w};
#pragma unroll
      for (int q = 0; q < 4; q++) {
        uint32_t e4l, e4h;
        nvfp4_e2m1x8_to_e4m3(ws[q], e4l, e4h);
        const __half2 rq = (q < 2) ? r0 : r1;  // q=0,1 -> blok 2s; q=2,3 -> 2s+1
        e4l = nvfp4_requant4(e4l, rq);
        e4h = nvfp4_requant4(e4h, rq);
        // przeplot z powrotem: bajt pary j sklada dim2j(lo) i dim2j+1(hi)
        // e4l = dimy parzyste (4B), e4h = nieparzyste; out potrzebuje
        // [d0,d1,d2,d3][d4..] — przeplatamy przez byte_perm.
        out[i][2 * q] = __byte_perm(e4l, e4h, 0x5140u);      // d0 d1 d2 d3
        out[i][2 * q + 1] = __byte_perm(e4l, e4h, 0x7362u);  // d4 d5 d6 d7
      }

      // rope: grupa 4 elemow nr s tokenu cand (16 grup pokrywa 64 elemy)
      if (rope_smem != nullptr) {
        const uint8_t* rslot = rope_smem + (size_t)cand * 128;
        const uint32_t rw =
            *reinterpret_cast<const uint32_t*>(rslot + NVFP4_ROPE_LANDING_OFF + 4 * s);
        nvfp4_fp8x4_to_bf16x4(rw, rope_out[i][0], rope_out[i][1]);
      }
    }
    bar();
    // --- faza write ---
#pragma unroll
    for (int i = 0; i < SEG_PER_THREAD; i++) {
      const int seg = phase * SEGS_PER_PHASE + i * THREADS + tid;
      const int cand = seg >> 4;
      const int s = seg & 15;
      uint8_t* slot = kv_smem + (size_t)cand * smem_stride;
      uint2* dst = reinterpret_cast<uint2*>(slot + 32 * s);
#pragma unroll
      for (int q = 0; q < 4; q++) dst[q] = make_uint2(out[i][2 * q], out[i][2 * q + 1]);
      if ((s & 3) == 0) {
        reinterpret_cast<float*>(slot + 512)[s >> 2] = tile_sc[i];
      }
      if (rope_smem != nullptr) {
        uint2* rdst = reinterpret_cast<uint2*>(rope_smem + (size_t)cand * 128 + 8 * s);
        *rdst = make_uint2(rope_out[i][0], rope_out[i][1]);
      }
    }
    bar();
  }
}
