# glm53-stack: vLLM serving image for GLM-5.3 (ModelOpt NVFP4, incoai/GLM-5.3-NVFP4) on 8x RTX PRO 6000 Blackwell (SM120, PCIe, no NVLink).
#
# Layering (bottom -> top):
#   [upstream]  vllm/vllm-openai:nightly-2afa3f7e950264bb179d030c23a1ed1f46558fd9  (vLLM 0.23.1rc1.dev925+g2afa3f7e9, 2026-07-08,
#               torch 2.11.0+cu130, FlashInfer 0.6.14).  !! This tag is no longer on Docker Hub (nightlies are pruned). !!
#   [team]      vllm-nightly-fi614:nvfp4 (image ID ccb91de90880): NVFP4 KV cache for sparse-MLA (patch_flashinfer.py, patch_vllm.py,
#               nvfp4_cache_ext.cu -> /opt/nvfp4ext), DCP for SM120 (dcp_sm120.patch), baked JIT sparse-MLA.  Exact steps: patches/team/BASE-BUILD-STEPS.txt.
#               Not rebuildable from public sources any more -> pinned by image ID; archived as vllm-nightly-fi614_nvfp4_ccb91de90880.tar.zst (see docs/RUNBOOK.md §1).
#   [this file] custom-AR PCIe patch (disabled by default), FlashInfer 0.6.18.post1, our vLLM patches (PP+MTP, fairness, online FP8/NVFP4, B12X hybrid).
#
# Build:   docker build -t glm53-stack:<YYYY.MM.DD> image/        (~15 min: pip download ~2 GB + JIT bake; needs internet: pypi + flashinfer.ai)
# Verify:  bench/run_all.sh (reproducibility gate: pip list, sha256 of patched files, KV pool, decode ms/step, parity, CE) - see docs/RUNBOOK.md
ARG BASE=vllm-nightly-fi614:nvfp4
FROM ${BASE}
ARG FI_VER=0.6.18.post1

# 0) [team, from nvfp4-car] custom all-reduce over PCIe (IPC/P2P one-shot).  Left in for parity with the validated image but DISABLED:
#    for >2 GPUs without NVLink it produces wrong results; the patch's own default is 2 MiB, hence the ENV below.
COPY patches/team/patch_custom_ar_pcie.py /patch/patch_custom_ar_pcie.py
RUN python3 /patch/patch_custom_ar_pcie.py
ENV VLLM_CUSTOM_AR_PCIE_MAX_SIZE=0

# 1) FlashInfer python + cubin + AOT jit-cache (cu130) in one consistent version. torch/triton/vLLM untouched (asserted).
#    nvidia-cutlass-dsl and apache-tvm-ffi pinned to the versions shipped by vllm/vllm-openai:nightly (09.2026).
RUN pip install --no-cache-dir --extra-index-url https://flashinfer.ai/whl/ --extra-index-url https://flashinfer.ai/whl/cu130/ \
      "flashinfer-python==${FI_VER}" "flashinfer-cubin==${FI_VER}" "flashinfer-jit-cache==${FI_VER}" \
      "nvidia-cutlass-dsl[cu13]==4.7.1" "apache-tvm-ffi==0.1.11" \
 && python3 -c "import torch, triton, flashinfer; assert torch.__version__.startswith('2.11.0'), torch.__version__; print('torch', torch.__version__, 'triton', triton.__version__, 'flashinfer', flashinfer.__version__)" \
 && pip list 2>/dev/null | grep -iE "flashinfer|tvm-ffi|cutlass-dsl|^torch "

# 2) jit-cache 0.6.18 ships a prebuilt sparse_mla_sm120 (AOT takes precedence over JIT) -> remove it so the team's patched
#    sources (ModelType GLM_NSA_NVFP4, 352 B/token) are used.
RUN d="$(python3 -c 'import flashinfer_jit_cache,os;print(os.path.dirname(flashinfer_jit_cache.__file__))')/jit_cache" \
 && test -d "$d/sparse_mla_sm120" && rm -rf "$d/sparse_mla_sm120" && echo "AOT sparse_mla_sm120 removed; $(ls "$d" | wc -l) modules left"

# 3) re-apply the team's FlashInfer patch (KV-NVFP4) to the 0.6.18 sources - 25/25 anchors verified (image/tools/check_anchors.py)
RUN python3 /patch/patch_flashinfer.py

# 4) bake JIT sparse-MLA SM120 into the arch dir "120f" = the one the runtime looks up on RTX PRO 6000
#    (without FLASHINFER_CUDA_ARCH_LIST a GPU-less build lands in cached_ops/ without arch and the runtime recompiles at every start).
RUN FLASHINFER_CUDA_ARCH_LIST="12.0f" python3 -c "from flashinfer.jit.mla import gen_sparse_mla_sm120_module; gen_sparse_mla_sm120_module().build(True); print('JIT-BAKED')" \
 && ls -la /root/.cache/flashinfer/${FI_VER}/120f/cached_ops/sparse_mla_sm120/

# 5) sanity: KV-NVFP4 loader + vLLM backends that touch FlashInfer
RUN python3 -c "import vllm_nvfp4_cache; print('LOADER-OK')" \
 && python3 -c "import vllm.v1.attention.backends.mla.flashinfer_mla_sparse_sm120, vllm.model_executor.layers.fused_moe.experts.flashinfer_b12x_moe; print('VLLM-IMPORT-OK')"

# 6) quack-kernels 0.5.0 (pinned to cutlass-dsl 4.5) breaks on 4.7.1 ("cute.core has no attribute ThrMma"); imported by
#    vLLM's dcp_indexer_cutedsl.py (indexer top-k merge for DCP>1). 0.6.5 = version shipped by vllm/vllm-openai:nightly.
RUN pip install --no-cache-dir --no-deps "quack-kernels==0.6.5" \
 && python3 -c "from quack.compile_utils import make_fake_tensor; import vllm.model_executor.kernels.attention.dsa.dcp_indexer_cutedsl; print('QUACK-OK')"

# 7) B12X with MTP (unquantized draft MoE -> auto) + B12X/CUTLASS hybrid gated by VLLM_B12X_HYBRID_MAX_TOKENS (off by default; negative result at MTP5)
COPY patches/ours/patch_b12x_hybrid.py /patch/patch_b12x_hybrid.py
RUN python3 /patch/patch_b12x_hybrid.py \
 && python3 -c "import vllm.model_executor.layers.fused_moe.experts.flashinfer_b12x_moe, vllm.model_executor.layers.fused_moe.oracle.unquantized; print('B12X-PATCH-IMPORT-OK')"

# 8) fairness: --long-prefill-token-threshold applied only when >1 request (VLLM_LONG_PREFILL_THRESHOLD_SHARED_ONLY=1 default)
COPY patches/ours/patch_sched_fair.py /patch/patch_sched_fair.py
RUN python3 /patch/patch_sched_fair.py && python3 -c "import vllm.v1.core.sched.scheduler; print('SCHED-PATCH-IMPORT-OK')"

# 9) MTP/EAGLE draft under pipeline parallelism (draft on the last stage; async scheduling required)
COPY patches/ours/patch_pp_mtp.py /patch/patch_pp_mtp.py
RUN python3 /patch/patch_pp_mtp.py && python3 -c "import vllm.config.speculative; print('PP-MTP-PATCH-IMPORT-OK')"

# 10) online FP8 W8A8 for BF16 linear layers excluded from quantization in the checkpoint (VLLM_TQ_FP8_LINEARS; off by default)
COPY patches/ours/patch_fp8_excluded.py /patch/patch_fp8_excluded.py
RUN python3 /patch/patch_fp8_excluded.py && python3 -c "import vllm.model_executor.layers.quantization.modelopt; print('FP8-PATCH-IMPORT-OK')"

# 11) MTP draft experts -> online NVFP4 (VLLM_TQ_NVFP4_MOE), per-channel FP8 (VLLM_TQ_FP8_CHANNEL=1), draft eh_proj as ReplicatedLinear
COPY patches/ours/patch_b1s2.py /patch/patch_b1s2.py
RUN python3 /patch/patch_b1s2.py \
 && python3 -c "import vllm.model_executor.layers.quantization.modelopt, vllm.model_executor.models.deepseek_mtp; print('B1S2-PATCH-IMPORT-OK')"

LABEL org.opencontainers.image.title="glm53-stack" \
      org.opencontainers.image.description="vLLM (nightly 2afa3f7e9) + KV-NVFP4 sparse-MLA + DCP SM120 + FlashInfer 0.6.18 + PP/MTP, fairness, online FP8/NVFP4 patches for GLM-5.3 NVFP4 on 8x RTX PRO 6000" \
      org.opencontainers.image.licenses="Apache-2.0"
