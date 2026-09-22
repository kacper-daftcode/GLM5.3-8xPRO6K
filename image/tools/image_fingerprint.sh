#!/usr/bin/env bash
# Fingerprint of a glm53-stack image (run INSIDE the image, no GPU needed):
#   docker run --rm --entrypoint bash -v $PWD/image/tools:/t:ro <image> /t/image_fingerprint.sh > fp.txt
# Prints pinned package versions and sha256 of every file touched by the patch layers, so two builds can be diffed.
set -uo pipefail
SP=/usr/local/lib/python3.12/dist-packages
echo "## packages"
pip list 2>/dev/null | grep -iE "^(vllm|torch|triton|flashinfer[a-z-]*|nvidia-cutlass-dsl|apache-tvm-ffi|quack-kernels|transformers|xformers) " | sort
echo "## patched vllm files"
for f in config/speculative.py config/vllm.py distributed/device_communicators/custom_all_reduce.py \
         model_executor/layers/attention/mla_attention.py model_executor/layers/quantization/modelopt.py \
         model_executor/models/deepseek_mtp.py model_executor/layers/fused_moe/experts/flashinfer_b12x_moe.py \
         model_executor/layers/fused_moe/oracle/unquantized.py v1/core/sched/scheduler.py v1/kv_cache_interface.py \
         v1/attention/backends/mla/flashinfer_mla_sparse.py v1/attention/backends/mla/flashinfer_mla_sparse_sm120.py \
         v1/attention/backends/mla/indexer.py v1/spec_decode/llm_base_proposer.py v1/worker/gpu_model_runner.py v1/worker/gpu_worker.py; do
  printf "%s  %s\n" "$(sha256sum "$SP/vllm/$f" | cut -c1-16)" "vllm/$f"
done
echo "## patched flashinfer files"
FI=$(python3 -c 'import flashinfer,os;print(os.path.dirname(flashinfer.__file__))' 2>/dev/null)
for f in $(cd "$FI" && ls data/include/flashinfer/attention/sparse_mla_sm120/*.cuh data/include/flashinfer/attention/sparse_mla_sm120/*/*.cuh data/include/flashinfer/attention/sparse_mla_sm120/*/*.h data/csrc/sparse_mla_sm120*.cu mla/_core.py mla/_sparse_mla_sm120.py 2>/dev/null | sort); do
  printf "%s  %s\n" "$(sha256sum "$FI/$f" | cut -c1-16)" "flashinfer/$f"
done
echo "## extras"
ls "$SP" | grep -E "vllm_nvfp4_cache" ; ls /opt/nvfp4ext/nvfp4_cache_ext/*.so 2>/dev/null | xargs -n1 basename
ls /root/.cache/flashinfer/*/120f/cached_ops/sparse_mla_sm120/ 2>/dev/null | grep -E "\.so$"
d="$(python3 -c 'import flashinfer_jit_cache,os;print(os.path.dirname(flashinfer_jit_cache.__file__))' 2>/dev/null)/jit_cache"; echo "jit_cache modules: $(ls "$d" 2>/dev/null | wc -l), sparse_mla_sm120 AOT present: $(test -d "$d/sparse_mla_sm120" && echo yes || echo no)"
echo "ENV VLLM_CUSTOM_AR_PCIE_MAX_SIZE=${VLLM_CUSTOM_AR_PCIE_MAX_SIZE:-unset}"
