"""Dump sygnatur wybranych funkcji/klas FlashInfer uzywanych przez vLLM (do diffu 0.6.14 vs 0.6.18)."""
import importlib, inspect, sys

TARGETS = [
    "flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla",
    "flashinfer.decode.trtllm_batch_decode_with_kv_cache",
    "flashinfer.prefill.trtllm_batch_context_with_kv_cache",
    "flashinfer.prefill.trtllm_ragged_attention_deepseek",
    "flashinfer.mla.sparse_mla_sm120",
    "flashinfer.mla._sparse_mla_sm120.sparse_mla_sm120",
    "flashinfer.mla._core.sparse_mla_sm120",
    "flashinfer.mla.trtllm_batch_decode_with_kv_cache_mla",
    "flashinfer.fused_moe.cutlass_fused_moe",
    "flashinfer.fused_moe.trtllm_fp4_block_scale_moe",
    "flashinfer.fused_moe.trtllm_fp4_block_scale_routed_moe",
    "flashinfer.fused_moe.B12xMoEWrapper",
    "flashinfer.fused_moe.B12xMoEWrapper.__init__",
    "flashinfer.fused_moe.B12xMoEWrapper.run",
    "flashinfer.fused_moe.B12xMoEWrapper.forward",
    "flashinfer.fused_moe.core.ActivationType",
    "flashinfer.fused_moe.core.get_w2_permute_indices_with_cache",
    "flashinfer.fused_moe.convert_to_block_layout",
    "flashinfer.fused_moe.WeightLayout",
    "flashinfer.fused_moe.Fp8QuantizationType",
    "flashinfer.mm_fp4",
    "flashinfer.bmm_fp8",
    "flashinfer.nvfp4_quantize",
    "flashinfer.nvfp4_block_scale_interleave",
    "flashinfer.block_scale_interleave",
    "flashinfer.fp4_quantization.nvfp4_block_scale_interleave",
    "flashinfer.cute_dsl.utils.convert_sf_to_mma_layout",
    "flashinfer.norm.rmsnorm",
    "flashinfer.norm.fused_add_rmsnorm",
    "flashinfer.rope.apply_rope_with_cos_sin_cache_inplace",
    "flashinfer.sampling.top_k_top_p_sampling_from_probs",
    "flashinfer.sampling.top_k_renorm_probs",
    "flashinfer.sampling.top_p_renorm_probs",
    "flashinfer.concat_ops.concat_mla_k",
    "flashinfer.autotuner.autotune",
    "flashinfer.utils.FP4Tensor",
    "flashinfer.RoutingMethodType",
    "flashinfer.SfLayout",
    "flashinfer.jit.mla.gen_sparse_mla_sm120_module",
    "flashinfer.comm.trtllm_allreduce_fusion",
    "flashinfer.comm.trtllm_create_ipc_workspace_for_all_reduce_fusion",
]


def resolve(path):
    parts = path.split(".")
    for i in range(len(parts), 0, -1):
        try:
            mod = importlib.import_module(".".join(parts[:i]))
        except Exception:
            continue
        obj = mod
        try:
            for p in parts[i:]:
                obj = getattr(obj, p)
            return obj
        except AttributeError:
            return None
    return None


import flashinfer
print("flashinfer", flashinfer.__version__)
for t in TARGETS:
    obj = resolve(t)
    if obj is None:
        print(f"{t}: MISSING")
        continue
    try:
        if inspect.isclass(obj):
            if issubclass(obj, __import__("enum").Enum):
                print(f"{t}: enum {[m.name for m in obj]}")
            else:
                print(f"{t}: class{inspect.signature(obj)}")
        else:
            print(f"{t}: {inspect.signature(obj)}")
    except Exception as e:
        print(f"{t}: <no signature: {e}>")
