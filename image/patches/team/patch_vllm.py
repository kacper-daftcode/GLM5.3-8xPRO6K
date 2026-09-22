"""Patch vLLM: --kv-cache-dtype nvfp4 dla FLASHINFER_MLA_SPARSE_SM120.

- backend deklaruje obsluge nvfp4 + ksztalt cache (num_blocks, block, 352)
- MLAAttentionSpec.real_page_size_bytes: 352 B/token
- impl SM120: kv_scale_format nvfp4_b16 + zapis przez vllm_nvfp4_cache
Idempotentny, twarde anchory.
"""
import pathlib

V = pathlib.Path("/usr/local/lib/python3.12/dist-packages/vllm")


def patch(path, old, new, count=1):
    s = path.read_text()
    if new in s:
        return
    assert old in s, f"ANCHOR NOT FOUND in {path}:\n{old[:200]}"
    assert s.count(old) == count, f"anchor x{s.count(old)} != {count} in {path}"
    path.write_text(s.replace(old, new))
    print(f"patched: {path.name}: {old[:60]!r}...")


# ---- 1. backend: dtypes + shape --------------------------------------------
B = V / "v1/attention/backends/mla/flashinfer_mla_sparse.py"
patch(
    B,
    """class FlashInferMLASparseSM120Backend(_FlashInferMLASparseBackendBase):
    \"\"\"FlashInfer sparse MLA backend for SM120.\"\"\"

    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",""",
    """class FlashInferMLASparseSM120Backend(_FlashInferMLASparseBackendBase):
    \"\"\"FlashInfer sparse MLA backend for SM120.\"\"\"

    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "nvfp4",  # TQ_NVFP4
        "auto",""",
)
patch(
    B,
    """        if cache_dtype_str in ("auto", "fp8", "fp8_e4m3", "fp8_ds_mla"):
            # fp8_ds_mla packed layout: 512 NoPE + 16 scales + 128 RoPE.
            return (num_blocks, block_size, 656)
        return (num_blocks, block_size, head_size)""",
    """        if cache_dtype_str == "nvfp4":  # TQ_NVFP4: LAYOUT-KV-NVFP4.md V1
            return (num_blocks, block_size, 352)
        if cache_dtype_str in ("auto", "fp8", "fp8_e4m3", "fp8_ds_mla"):
            # fp8_ds_mla packed layout: 512 NoPE + 16 scales + 128 RoPE.
            return (num_blocks, block_size, 656)
        return (num_blocks, block_size, head_size)""",
)

# ---- 2. spec: rozmiar strony -----------------------------------------------
K = V / "v1/kv_cache_interface.py"
patch(
    K,
    """    @property
    def real_page_size_bytes(self) -> int:
        if self.cache_dtype_str == "fp8_ds_mla":
            if self.model_version == "deepseek_v4":
                # DeepseekV4: 448B NoPE + 128B RoPE + 8B fp8 scale = 584B per token.
                # head_size stays semantic (512); bytes are determined here.
                return self.storage_block_size * 584
            # V3.2 main MLA: 656-byte custom layout (kv_lora_rank=512 +
            # qk_rope_head_dim=64, head_size=576). See flashmla_sparse.py.
            return self.block_size * 656""",
    """    @property
    def real_page_size_bytes(self) -> int:
        if self.cache_dtype_str == "nvfp4":
            # TQ_NVFP4: 256B E2M1 + 32B skale E4M3 + 64B rope FP8 = 352 B/token
            return self.block_size * 352
        if self.cache_dtype_str == "fp8_ds_mla":
            if self.model_version == "deepseek_v4":
                # DeepseekV4: 448B NoPE + 128B RoPE + 8B fp8 scale = 584B per token.
                # head_size stays semantic (512); bytes are determined here.
                return self.storage_block_size * 584
            # V3.2 main MLA: 656-byte custom layout (kv_lora_rank=512 +
            # qk_rope_head_dim=64, head_size=576). See flashmla_sparse.py.
            return self.block_size * 656""",
)

# ---- 3. impl SM120: dtype + kv_scale_format + zapis --------------------------
I = V / "v1/attention/backends/mla/flashinfer_mla_sparse_sm120.py"
patch(
    I,
    """        self.kv_cache_dtype = kv_cache_dtype
        if self.kv_cache_dtype != "fp8_ds_mla":
            raise NotImplementedError(
                "FLASHINFER_MLA_SPARSE_SM120 requires the packed fp8_ds_mla "
                f"KV cache layout; got kv_cache_dtype={kv_cache_dtype!r}."
            )""",
    """        self.kv_cache_dtype = kv_cache_dtype
        if self.kv_cache_dtype not in ("fp8_ds_mla", "nvfp4"):  # TQ_NVFP4
            raise NotImplementedError(
                "FLASHINFER_MLA_SPARSE_SM120 requires the packed fp8_ds_mla "
                f"or nvfp4 KV cache layout; got kv_cache_dtype={kv_cache_dtype!r}."
            )""",
)
patch(
    I,
    """        self.kv_scale_format = _kv_scale_format_for_model(model_type)""",
    """        if self.kv_cache_dtype == "nvfp4":  # TQ_NVFP4
            self.kv_scale_format = "nvfp4_b16"
        else:
            self.kv_scale_format = _kv_scale_format_for_model(model_type)""",
)
patch(
    I,
    """        self.supports_quant_query_input = False
        self._workspace_buffer: torch.Tensor | None = None""",
    """        self.supports_quant_query_input = False
        self._workspace_buffer: torch.Tensor | None = None

    # TQ_NVFP4: zapis KV w layoucie nvfp4_ds_mla (352 B/token) przez
    # rozszerzenie vllm_nvfp4_cache; pozostale dtype'y jak w klasie bazowej.
    def do_kv_cache_update(
        self,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        kv_cache_dtype: str,
        k_scale: torch.Tensor,
    ) -> None:
        if kv_cache.numel() == 0:
            return
        if kv_cache_dtype == "nvfp4":
            import vllm_nvfp4_cache

            k_pe_2d = k_pe.squeeze(1) if k_pe.dim() == 3 else k_pe
            vllm_nvfp4_cache.concat_and_cache_nvfp4_ds_mla(
                kv_c_normed, k_pe_2d, kv_cache, slot_mapping.flatten()
            )
            return
        super().do_kv_cache_update(
            kv_c_normed, k_pe, kv_cache, slot_mapping, kv_cache_dtype, k_scale
        )""",
)

# ---- 3b. supports_combination: druga twarda lista dtype'ow ------------------
patch(
    B,
    """        if dtype != torch.bfloat16:
            return "dtype not supported"
        if kv_cache_dtype not in (
            None,
            "auto",
            "fp8",
            "fp8_e4m3",
            "fp8_ds_mla",
        ):
            return "kv_cache_dtype not supported\"""",
    """        if dtype != torch.bfloat16:
            return "dtype not supported"
        if kv_cache_dtype not in (
            None,
            "auto",
            "fp8",
            "fp8_e4m3",
            "fp8_ds_mla",
            "nvfp4",  # TQ_NVFP4
        ):
            return "kv_cache_dtype not supported\"""",
)

# ---- 3c. spec MLA musi niesc kv_quant_mode=NVFP4 -----------------------------
# gpu_model_runner wybiera cache_dtype_str="auto" (=656 B) gdy kv_quant_mode
# spec-a to NONE; dla nvfp4 ksztalt ma byc 352 B.
M = V / "model_executor/layers/attention/mla_attention.py"
patch(
    M,
    """        return MLAAttentionSpec(
            block_size=vllm_config.cache_config.block_size,
            num_kv_heads=1,
            head_size=self.head_size,
            dtype=kv_cache_dtype,
            cache_dtype_str=vllm_config.cache_config.cache_dtype,
        )""",
    """        from vllm.v1.kv_cache_interface import KVQuantMode, get_kv_quant_mode

        return MLAAttentionSpec(
            block_size=vllm_config.cache_config.block_size,
            num_kv_heads=1,
            head_size=self.head_size,
            dtype=kv_cache_dtype,
            cache_dtype_str=vllm_config.cache_config.cache_dtype,
            # TQ_NVFP4: bez tego gpu_model_runner bierze ksztalt "auto" (656 B)
            kv_quant_mode=(
                get_kv_quant_mode(vllm_config.cache_config.cache_dtype)
                if vllm_config.cache_config.cache_dtype == "nvfp4"
                else KVQuantMode.NONE
            ),
        )""",
)

# ---- 4. zdejmij globalna blokade nvfp4+MLA (mamy wlasny backend SM120) ------
C = V / "config/vllm.py"
patch(
    C,
    """        if self.cache_config.cache_dtype == "nvfp4" and self.model_config.use_mla:
            raise ValueError(
                "nvfp4 KV cache is not supported with MLA (Multi-head Latent "
                "Attention) backends. Please use a different --kv-cache-dtype "
                "(e.g., 'fp8' or 'auto') for MLA models such as DeepSeek."
            )""",
    """        # TQ_NVFP4: sparse-MLA SM120 ma wlasny layout nvfp4_ds_mla (352 B/tok);
        # blokada zostaje dla pozostalych platform/backendow MLA.
        from vllm.platforms import current_platform

        if (
            self.cache_config.cache_dtype == "nvfp4"
            and self.model_config.use_mla
            and not current_platform.is_device_capability_family(120)
        ):
            raise ValueError(
                "nvfp4 KV cache is not supported with MLA (Multi-head Latent "
                "Attention) backends. Please use a different --kv-cache-dtype "
                "(e.g., 'fp8' or 'auto') for MLA models such as DeepSeek."
            )""",
)

import ast

for f in (B, K, I, C, M):
    ast.parse(f.read_text())
print("SYNTAX-OK")
print("PATCH-VLLM-DONE")
