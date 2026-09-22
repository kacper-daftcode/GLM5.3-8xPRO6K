#!/usr/bin/env bash
# Start GLM-5.3 NVFP4 on an UNPATCHED upstream vLLM image (e.g. vllm/vllm-openai:nightly) on the testbed — for reproducing upstream issues
# (plan 4.1: MTP acceptance under pipeline parallelism) and for the port to current vLLM. Functional settings, not the production ones:
# no DCP (upstream has no DCP for the SM120 sparse-MLA backend), KV dtype selectable, short context, few sequences.
# Knobs: IMAGE TP PP MTP ASYNC KV MAX_LEN MAX_SEQS BATCHED EXTRA NAME CACHE; MODEL/CACHE_ROOT/GPUS from bench/env.
set -euo pipefail
[ -f "$(dirname "$0")/env" ] && source "$(dirname "$0")/env"
GPUS=${GPUS:-0,1,2,3,4,5,7,8}
IMAGE=${IMAGE:-vllm/vllm-openai:nightly}
MODEL=${MODEL:?set MODEL (checkpoint directory) in bench/env or the environment}
CACHE_ROOT=${CACHE_ROOT:-/var/cache/glm53-stack}
TP=${TP:-8}; PP=${PP:-1}; MTP=${MTP:-3}; ASYNC=${ASYNC:-1}
KV=${KV:-fp8_ds_mla}                 # upstream KV dtype for DeepSeek-style sparse MLA; fall back to auto if rejected
MAX_LEN=${MAX_LEN:-131072}; MAX_SEQS=${MAX_SEQS:-8}; BATCHED=${BATCHED:-2048}
EXTRA=${EXTRA:-}
VOLX=${VOLX:-}                       # extra volumes "host:container[:ro] ...", e.g. the real checkpoint dir when MODEL is an overlay of symlinks
NAME=${NAME:-upstream-glm}
CACHE=${CACHE:-$CACHE_ROOT/upstream-$(echo "$IMAGE" | tr '/:' '__')}
docker rm -f "$NAME" >/dev/null 2>&1 || true
ARGS=(/model --served-model-name glm-5.3 --tensor-parallel-size "$TP" --pipeline-parallel-size "$PP"
      --quantization modelopt_fp4 --kv-cache-dtype "$KV" --trust-remote-code --gpu-memory-utilization 0.90
      --reasoning-parser glm45 --tool-call-parser glm47 --enable-auto-tool-choice
      --max-model-len "$MAX_LEN" --max-num-seqs "$MAX_SEQS" --max-num-batched-tokens "$BATCHED" --host 0.0.0.0 --port 8000)
if [ "$ASYNC" = 1 ]; then ARGS+=(--async-scheduling); else ARGS+=(--no-async-scheduling); fi
[ "$MTP" -gt 0 ] && ARGS+=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$MTP}")
# shellcheck disable=SC2206
[ -n "$EXTRA" ] && ARGS+=($EXTRA)
VOLS=(); for v in $VOLX; do VOLS+=(-v "$v"); done
mkdir -p "$CACHE"
echo "[$(date -u +%T)] start $NAME: image=$IMAGE TP=$TP PP=$PP MTP=$MTP ASYNC=$ASYNC KV=$KV max-len=$MAX_LEN model=$MODEL cache=$CACHE"
docker run -d --name "$NAME" --runtime nvidia --gpus "\"device=$GPUS\"" --network host --ipc host --shm-size 32g --ulimit memlock=-1:-1 \
  -e NCCL_P2P_DISABLE=0 -e NCCL_P2P_LEVEL=SYS -e VLLM_ENGINE_READY_TIMEOUT_S=3600 \
  -v "$MODEL":/model -v "$CACHE":/root/.cache/vllm "${VOLS[@]}" "$IMAGE" "${ARGS[@]}" >/dev/null
for i in $(seq 1 360); do
  if curl -s -m 3 -o /dev/null -w '%{http_code}' http://127.0.0.1:8000/health | grep -q 200; then echo "[$(date -u +%T)] READY po $((i*5)) s"; exit 0; fi
  st=$(docker inspect -f '{{.State.Status}}' "$NAME" 2>/dev/null || echo missing)
  if [ "$st" != running ]; then echo "[$(date -u +%T)] KONTENER PADL ($st):"; docker logs --tail 40 "$NAME" 2>&1 | grep -v '\^\^\^' | tail -25 | cut -c1-240; exit 1; fi
  sleep 5
done
echo TIMEOUT; exit 1
