#!/usr/bin/env bash
# Qwen3.8-Flash-Next FP8 — dedykowany obraz z recipe vLLM. Na 8 GPU wymagane TEP8 (czyste TP8 niezgodne z blokami FP8 128x128).
set -euo pipefail
source "$(dirname "$(readlink -f "$0")")/../common.sh"   # deploy/common.sh (reads deploy/env); works through a symlink
MODEL_DIR=${MODEL_DIR:?set MODEL_DIR (Qwen3.8-Flash-Next-FP8 checkpoint) in deploy/env or the environment}
EXTRA_ARGS=${EXTRA_ARGS:-}          # np. "--moe-backend triton"
PLE_OFFLOAD=${PLE_OFFLOAD:-0}       # 1 => tabela n-gram (51B) w RAM hosta
stop_big_gpu_containers
echo "[$(date -u +%T)] start qwen38-flash-next-test z $MODEL_DIR (PLE_OFFLOAD=$PLE_OFFLOAD EXTRA_ARGS=$EXTRA_ARGS)"
# shellcheck disable=SC2086
sudo docker run --name qwen38-flash-next-test "${COMMON_DOCKER_ARGS[@]}" \
  -e VLLM_PLE_CPU_OFFLOAD="$PLE_OFFLOAD" \
  -v "$MODEL_DIR":/model \
  vllm/vllm-openai:qwen38-flash-next \
  --model /model --served-model-name qwen3.8-flash-next \
  --tensor-parallel-size 8 --enable-expert-parallel \
  --gpu-memory-utilization 0.90 --max-num-seqs 256 --enable-prefix-caching \
  --no-enable-flashinfer-autotune \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml --reasoning-parser qwen3 \
  $EXTRA_ARGS \
  --host 0.0.0.0 --port "$PORT"
wait_ready 2400 qwen38-flash-next-test
