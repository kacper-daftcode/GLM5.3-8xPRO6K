#!/usr/bin/env bash
# GLM-5.3-Flash natywne FP8 — dedykowany obraz z recipe vLLM (FlashInfer >= 0.6.17 dla NoPE sparse MLA)
set -euo pipefail
source "$(dirname "$(readlink -f "$0")")/../common.sh"   # deploy/common.sh (reads deploy/env); works through a symlink
MODEL_DIR=${MODEL_DIR:?set MODEL_DIR (GLM-5.3-Flash checkpoint) in deploy/env or the environment}
MAX_LEN=${MAX_LEN:-262144}
stop_big_gpu_containers
echo "[$(date -u +%T)] start glm53-flash-test z $MODEL_DIR"
sudo docker run --name glm53-flash-test "${COMMON_DOCKER_ARGS[@]}" \
  -v "$MODEL_DIR":/model \
  vllm/vllm-openai:glm53-flash \
  --model /model --served-model-name glm-5.3-flash \
  --tensor-parallel-size 8 --kv-cache-dtype fp8 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":5}' \
  --reasoning-parser glm45 --tool-call-parser glm47 --enable-auto-tool-choice \
  --no-enable-flashinfer-autotune \
  --gpu-memory-utilization 0.90 --max-model-len "$MAX_LEN" --max-num-seqs 64 \
  --host 0.0.0.0 --port "$PORT"
wait_ready 2400 glm53-flash-test
