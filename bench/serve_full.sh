#!/usr/bin/env bash
# Host B: PELNY GLM-5.3-NVFP4 (incoai) na 8 kartach PRO, konfiguracja 1:1 z prod A (do A/B bez dotykania prod).
# Pokretla przez env. Domyslnie = aktualny prod. Ustawienia hosta (MODEL, CACHE_ROOT, RESULTS_DIR, PROF_DIR, GPUS) z pliku "env"
# obok skryptu (git-ignored, wzor env.example) albo ze srodowiska.
set -euo pipefail
[ -f "$(dirname "$0")/env" ] && source "$(dirname "$0")/env"
GPUS=${GPUS:-0,1,2,3,4,5,7,8}        # 8x PRO 6000 (pozostale GPU hosta pomijamy)
TP=${TP:-8}; PP=${PP:-1}; DCP=${DCP:-2}; MTP=${MTP:-5}
MOE=${MOE:-auto}; KV=${KV:-nvfp4}
MAX_SEQS=${MAX_SEQS:-32}; MAX_LEN=${MAX_LEN:-750000}; BATCHED=${BATCHED:-2048}
NCCL_MODE=${NCCL_MODE:-p2p_sys}      # p2p_sys (prod) | shm
PROFILE=${PROFILE:-0}; EXTRA=${EXTRA:-}
ASYNC=${ASYNC:-1}                    # 1 = --async-scheduling (prod); 0 = bez (wymagane dla PP>1 z MTP w tym vLLM)
ENVX=${ENVX:-}                        # dodatkowe zmienne env do kontenera, np. ENVX="VLLM_B12X_HYBRID_MAX_TOKENS=80 FOO=1"
VOLX=${VOLX:-}                        # dodatkowe wolumeny, np. VOLX="/path/to/GLM-5.3-DFlash2:/draft"
DOCKERX=${DOCKERX:-}                  # dodatkowe argumenty docker run (debug), np. DOCKERX="--cap-add=SYS_PTRACE --security-opt seccomp=unconfined"
IMAGE=${IMAGE:-vllm-nightly-fi614:nvfp4-car}
MODEL=${MODEL:?set MODEL (checkpoint directory) in bench/env or the environment}
NAME=${NAME:-full-glm}
CACHE_ROOT=${CACHE_ROOT:-/var/cache/glm53-stack}
CACHE=${CACHE:-$CACHE_ROOT/vllm}        # trwaly cache torch.compile/Triton; osobny katalog per obraz/konfiguracja (np. $CACHE_ROOT/vllm-fi618)
RESULTS_DIR=${RESULTS_DIR:-$(cd "$(dirname "$0")/.." && pwd)/results}
PROF_DIR=${PROF_DIR:-$RESULTS_DIR/prof} # slady torch profilera (PROFILE=1) na hoscie
docker rm -f $NAME >/dev/null 2>&1 || true
ARGS=(--model /model --served-model-name glm-5.3 --tensor-parallel-size $TP --pipeline-parallel-size $PP --decode-context-parallel-size $DCP
      --quantization modelopt_fp4 --kv-cache-dtype $KV --trust-remote-code --gpu-memory-utilization 0.90
      --reasoning-parser glm45 --tool-call-parser glm47 --enable-auto-tool-choice
      --max-model-len $MAX_LEN --max-num-seqs $MAX_SEQS --max-num-batched-tokens $BATCHED
      --host 0.0.0.0 --port 8000)
if [ "$ASYNC" = 1 ]; then ARGS+=(--async-scheduling); else ARGS+=(--no-async-scheduling); fi
DRAFT_Q=${DRAFT_Q:-}                 # kwantyzacja drafta MTP online (np. fp8) - warstwa 78 w checkpoincie jest BF16
if [ "$MTP" -gt 0 ]; then if [ -n "$DRAFT_Q" ]; then ARGS+=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$MTP,\"quantization\":\"$DRAFT_Q\"}"); else ARGS+=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$MTP}"); fi; fi
[ "$MOE" != auto ] && ARGS+=(--moe-backend "$MOE")
[ -n "$EXTRA" ] && ARGS+=($EXTRA)
[ "$PROFILE" = 1 ] && ARGS+=(--profiler-config '{"profiler":"torch","torch_profiler_dir":"/prof","torch_profiler_with_stack":false}')
ENVS=(-e VLLM_CUSTOM_AR_PCIE_MAX_SIZE=0 -e FLASHINFER_DISABLE_VERSION_CHECK=1 -e VLLM_ENGINE_READY_TIMEOUT_S=3600 -e TRITON_CACHE_DIR=/root/.cache/vllm/triton)
if [ "$NCCL_MODE" = p2p_sys ]; then ENVS+=(-e NCCL_P2P_DISABLE=0 -e NCCL_P2P_LEVEL=SYS); else ENVS+=(-e NCCL_P2P_DISABLE=1); fi
for kv in $ENVX; do ENVS+=(-e "$kv"); done
for v in $VOLX; do ENVS+=(-v "$v"); done
mkdir -p "$CACHE" "$PROF_DIR"
echo "[$(date -u +%T)] start $NAME: GPUS=$GPUS TP=$TP PP=$PP DCP=$DCP MTP=$MTP ASYNC=$ASYNC MOE=$MOE KV=$KV NCCL=$NCCL_MODE batched=$BATCHED image=$IMAGE cache=$CACHE"
docker run -d --name $NAME --runtime nvidia --gpus "\"device=$GPUS\"" --network host --ipc host --shm-size 32g --ulimit memlock=-1:-1 $DOCKERX \
  "${ENVS[@]}" -v "$MODEL":/model -v "$PROF_DIR":/prof -v "$CACHE":/root/.cache/vllm "$IMAGE" "${ARGS[@]}" >/dev/null
for i in $(seq 1 360); do
  if curl -s -m 3 -o /dev/null -w '%{http_code}' http://127.0.0.1:8000/health | grep -q 200; then echo "[$(date -u +%T)] READY po $((i*5)) s"; exit 0; fi
  st=$(docker inspect -f '{{.State.Status}}' $NAME 2>/dev/null || echo missing)
  if [ "$st" != running ]; then echo "[$(date -u +%T)] KONTENER PADL ($st):"; docker logs --tail 30 $NAME 2>&1 | grep -v '\^\^\^' | tail -20 | cut -c1-220; exit 1; fi
  sleep 5
done
echo TIMEOUT; exit 1
