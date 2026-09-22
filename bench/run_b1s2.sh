#!/usr/bin/env bash
# B1 etap 2 (test przez bind-mount spatchowanych plikow; obraz fi618 bez zmian).
# PATCHED_DIR (bench/env) = katalog z modelopt.py i deepseek_mtp.py po nalozeniu patch_fp8_excluded.py + patch_b1s2.py.
set -euo pipefail
[ -f "$(dirname "$0")/env" ] && source "$(dirname "$0")/env"
CACHE_ROOT=${CACHE_ROOT:-/var/cache/glm53-stack}
PATCHED_DIR=${PATCHED_DIR:?set PATCHED_DIR (directory with patched modelopt.py / deepseek_mtp.py) in bench/env or the environment}
SP=/usr/local/lib/python3.12/dist-packages/vllm
FP8=${FP8:-fused_qkv_a_proj,q_b_proj,o_proj,shared_experts,indexer.wq_b,lm_head,eh_proj}
PART=${PART:-42,36}
CH=${CH:-1}
DRAFT=${DRAFT:-layers.78.mlp.experts}
MARGIN=${MARGIN:-2.0}
EXTRA_ENV=${EXTRA_ENV:-}
CACHE=${CACHE:-$CACHE_ROOT/vllm-fi618-b1s2}
IMAGE=vllm-nightly-fi614:nvfp4-fi618 CACHE=$CACHE TP=4 PP=2 DCP=1 MTP=5 ASYNC=1 \
VOLX="$PATCHED_DIR/modelopt.py:$SP/model_executor/layers/quantization/modelopt.py $PATCHED_DIR/deepseek_mtp.py:$SP/model_executor/models/deepseek_mtp.py" \
ENVX="VLLM_PP_LAYER_PARTITION=$PART VLLM_TQ_FP8_LINEARS=$FP8 VLLM_TQ_FP8_CHANNEL=$CH VLLM_TQ_NVFP4_MOE=$DRAFT VLLM_TQ_NVFP4_MOE_AMAX_MARGIN=$MARGIN $EXTRA_ENV" \
EXTRA="--long-prefill-token-threshold 512" "$(dirname "$0")/serve_full.sh"
