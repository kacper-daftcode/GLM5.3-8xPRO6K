// concat_and_cache_nvfp4_ds_mla: zapis latentu MLA + rope do KV cache w
// layoucie nvfp4_ds_mla V1 (352 B/token, LAYOUT-KV-NVFP4.md §2-3).
//
// Odpowiednik concat_and_cache_ds_mla_kernel z vLLM (cache_kernels.cu),
// ale kwantyzacja NVFP4 blok-16 zamiast FP8 tile-128. Semantyka §3:
// dokladnie div.rn (nie rcp.approx) — bit-exact z nvfp4_pack.pack_kv_nvfp4
// (weryfikacja: test_write_op.py).
//
// Uwaga: k_scale checkpointu ignorowany (jak w prod fp8_ds_mla); skala
// globalna to stala 2^-6 wspoldzielona z kernelem odczytu.

#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

namespace {

constexpr float kGlobalScale = 0.015625f;   // 2^-6
constexpr float kInvGlobalScale = 64.0f;
constexpr float kE2M1Max = 6.0f;
constexpr int kLatentDim = 512;
constexpr int kRopeDim = 64;
constexpr int kScaleOff = 256;
constexpr int kRopeOff = 288;
constexpr int kStride = 352;

__device__ __forceinline__ uint32_t fp32x8_to_e2m1x8(const float f[8]) {
  uint32_t val;
  asm volatile(
      "{\n"
      ".reg .b8 b0, b1, b2, b3;\n"
      "cvt.rn.satfinite.e2m1x2.f32 b0, %2, %1;\n"
      "cvt.rn.satfinite.e2m1x2.f32 b1, %4, %3;\n"
      "cvt.rn.satfinite.e2m1x2.f32 b2, %6, %5;\n"
      "cvt.rn.satfinite.e2m1x2.f32 b3, %8, %7;\n"
      "mov.b32 %0, {b0, b1, b2, b3};\n"
      "}"
      : "=r"(val)
      : "f"(f[0]), "f"(f[1]), "f"(f[2]), "f"(f[3]), "f"(f[4]), "f"(f[5]),
        "f"(f[6]), "f"(f[7]));
  return val;
}

// Blok = 64 watki (2 warpy): warp 0 -> latent (32 blokow-16), warp 1 -> rope.
__global__ void concat_and_cache_nvfp4_kernel(
    const __nv_bfloat16* __restrict__ kv_c,  // [T, 512]
    const __nv_bfloat16* __restrict__ k_pe,  // [T, 64]
    uint8_t* __restrict__ kv_cache,          // [num_blocks, page, 352]
    const int64_t* __restrict__ slot_mapping,  // [T]
    const int64_t kv_c_stride, const int64_t k_pe_stride,
    const int64_t block_stride_bytes, const int page_size) {
  const int64_t token_idx = blockIdx.x;
  const int64_t slot_idx = slot_mapping[token_idx];
  if (slot_idx < 0) return;  // token wypelniajacy (padding)
  const int64_t block_idx = slot_idx / page_size;
  const int64_t block_off = slot_idx % page_size;
  uint8_t* dst =
      kv_cache + block_idx * block_stride_bytes + block_off * (int64_t)kStride;

  const int lane = threadIdx.x & 31;

  if (threadIdx.x < 32) {
    // --- latent: 1 blok-16 na lane ---
    const __nv_bfloat16* src = kv_c + token_idx * kv_c_stride + lane * 16;
    float x[16];
#pragma unroll
    for (int i = 0; i < 16; i++) x[i] = __bfloat162float(src[i]);

    float amax = 0.f;
#pragma unroll
    for (int i = 0; i < 16; i++) amax = fmaxf(amax, fabsf(x[i]));

    const float sf = amax / kE2M1Max;                      // div.rn
    const __nv_fp8_e4m3 sf_q(fminf(sf * kInvGlobalScale, 448.0f));
    const float scale = float(sf_q) * kGlobalScale;        // dokladne (2^-6)
    const float inv = (scale > 0.f) ? (1.0f / scale) : 0.f;  // div.rn

    float q[16];
#pragma unroll
    for (int i = 0; i < 16; i++) q[i] = x[i] * inv;

    uint2 packed;
    packed.x = fp32x8_to_e2m1x8(q);
    packed.y = fp32x8_to_e2m1x8(q + 8);
    *reinterpret_cast<uint2*>(dst + lane * 8) = packed;
    dst[kScaleOff + lane] = *reinterpret_cast<const uint8_t*>(&sf_q);
  } else if (lane < 16) {
    // --- rope: 4 elementy na lane, E4M3 skala 1.0 ---
    const __nv_bfloat16* src = k_pe + token_idx * k_pe_stride + lane * 4;
    uint32_t out = 0;
#pragma unroll
    for (int i = 0; i < 4; i++) {
      const float r =
          fminf(fmaxf(__bfloat162float(src[i]), -448.0f), 448.0f);
      const __nv_fp8_e4m3 rq(r);
      out |= uint32_t(*reinterpret_cast<const uint8_t*>(&rq)) << (8 * i);
    }
    *reinterpret_cast<uint32_t*>(dst + kRopeOff + lane * 4) = out;
  }
}

void concat_and_cache_nvfp4_ds_mla(torch::Tensor& kv_c, torch::Tensor& k_pe,
                                   torch::Tensor& kv_cache,
                                   torch::Tensor& slot_mapping) {
  TORCH_CHECK(kv_c.dtype() == torch::kBFloat16, "kv_c musi byc bf16");
  TORCH_CHECK(k_pe.dtype() == torch::kBFloat16, "k_pe musi byc bf16");
  TORCH_CHECK(kv_cache.dtype() == torch::kUInt8, "kv_cache musi byc uint8");
  TORCH_CHECK(slot_mapping.dtype() == torch::kInt64);
  TORCH_CHECK(kv_c.dim() == 2 && kv_c.size(-1) == kLatentDim, "kv_c [T,512]");
  TORCH_CHECK(k_pe.dim() == 2 && k_pe.size(-1) == kRopeDim, "k_pe [T,64]");
  TORCH_CHECK(kv_cache.size(-1) == kStride, "kv_cache ostatni wymiar 352");
  TORCH_CHECK(kv_c.stride(-1) == 1 && k_pe.stride(-1) == 1);

  const int num_tokens = slot_mapping.size(0);
  if (num_tokens == 0) return;
  const int page_size = kv_cache.size(1);

  const c10::cuda::OptionalCUDAGuard guard(kv_c.device());
  const cudaStream_t stream =
      c10::cuda::getCurrentCUDAStream(kv_c.device().index()).stream();
  concat_and_cache_nvfp4_kernel<<<num_tokens, 64, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(kv_c.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(k_pe.data_ptr()),
      kv_cache.data_ptr<uint8_t>(), slot_mapping.data_ptr<int64_t>(),
      kv_c.stride(0), k_pe.stride(0), kv_cache.stride(0), page_size);
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("concat_and_cache_nvfp4_ds_mla", &concat_and_cache_nvfp4_ds_mla,
        "Zapis KV MLA w layoucie nvfp4_ds_mla V1 (352 B/token)");
}
