#!/usr/bin/env bash
# GLM-5.3 NVFP4 (incoai) — te same flagi co prod GLM-5.2, ten sam obraz. Alias glm-5.2 zeby klienci nie padli.
set -euo pipefail
source "$(dirname "$(readlink -f "$0")")/../common.sh"   # deploy/common.sh (reads deploy/env); works through a symlink
MODEL_DIR=${MODEL_DIR:?set MODEL_DIR (GLM-5.3-NVFP4 checkpoint) in deploy/env or the environment}
stop_big_gpu_containers
echo "[$(date -u +%T)] start glm53-nvfp4-test z $MODEL_DIR"
sudo docker run --name glm53-nvfp4-test "${COMMON_DOCKER_ARGS[@]}" \
  -v "$MODEL_DIR":/model \
  vllm-nightly-fi614:nvfp4 \
  --model /model --served-model-name glm-5.3 glm-5.2 \
  --tensor-parallel-size 8 --decode-context-parallel-size 4 \
  --quantization modelopt_fp4 --kv-cache-dtype nvfp4 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --reasoning-parser glm45 --tool-call-parser glm47 --enable-auto-tool-choice \
  --trust-remote-code --gpu-memory-utilization 0.90 \
  --max-model-len 750000 --max-num-seqs 4 \
  --host 0.0.0.0 --port "$PORT"
wait_ready 1800 glm53-nvfp4-test
