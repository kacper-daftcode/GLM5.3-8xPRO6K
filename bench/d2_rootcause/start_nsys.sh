#!/usr/bin/env bash
# D2 debug launcher: production candidate config (mirrors bench/run_cand.sh defaults) run under Nsight Systems so that the
# CUDA API/kernel/memcpy timeline of every worker is captured; on hang send SIGINT to the container (nsys finalizes the report,
# --kill=sigkill terminates vLLM). Report: $CACHE/nsys_hang.nsys-rep on the host.
set -u
BENCH=$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)
[ -f "$BENCH/env" ] && source "$BENCH/env"                  # host settings (MODEL, CACHE_ROOT), see bench/env.example
KV_OFFLOAD_GB=${KV_OFFLOAD_GB:-32}
CACHE_ROOT=${CACHE_ROOT:-/var/cache/glm53-stack}
CACHE=${CACHE:-$CACHE_ROOT/vllm-fi618-b1s2-p41-mtp3}         # same compile cache as run_cand.sh for this config
MODEL=${MODEL:?set MODEL (checkpoint directory) in bench/env or the environment}
IMAGE=${IMAGE:-glm53-stack:2026.09.19-rc2-dbg2}
NSYS=/opt/nvidia/nsight-systems/2026.3.2/bin/nsys
KV_JSON="{\"kv_connector\":\"OffloadingConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"cpu_bytes_to_use\":$((KV_OFFLOAD_GB*1024*1024*1024))}}"
rm -f $CACHE/nsys_hang.nsys-rep $CACHE/nsys_hang.sqlite
docker rm -f full-glm >/dev/null 2>&1
docker run -d --name full-glm --runtime nvidia --gpus '"device=0,1,2,3,4,5,7,8"' --network host --ipc host --shm-size 32g --ulimit memlock=-1:-1 \
  --cap-add=SYS_PTRACE --cap-add=SYS_ADMIN --security-opt seccomp=unconfined \
  -e VLLM_CUSTOM_AR_PCIE_MAX_SIZE=0 -e FLASHINFER_DISABLE_VERSION_CHECK=1 -e VLLM_ENGINE_READY_TIMEOUT_S=3600 -e TRITON_CACHE_DIR=/root/.cache/vllm/triton \
  -e NCCL_P2P_DISABLE=0 -e NCCL_P2P_LEVEL=SYS \
  -e VLLM_PP_LAYER_PARTITION=41,37 -e VLLM_TQ_FP8_LINEARS=fused_qkv_a_proj,q_b_proj,o_proj,shared_experts,indexer.wq_b,lm_head,eh_proj -e VLLM_TQ_FP8_CHANNEL=1 \
  -e VLLM_TQ_NVFP4_MOE=layers.78.mlp.experts ${EXTRA_ENV:+$(for kv in $EXTRA_ENV; do printf -- "-e %s " "$kv"; done)} \
  -v "$MODEL":/model -v "$CACHE":/root/.cache/vllm \
  --entrypoint $NSYS "$IMAGE" profile --session-new=hang --cuda-flush-interval=2000 --trace=cuda --cuda-graph-trace=node --sample=none --cpuctxsw=none \
  --trace-fork-before-exec=true --kill=sigkill --output=/root/.cache/vllm/nsys_hang --force-overwrite=true -- \
  vllm serve ${VLLM_EXTRA_ARGS:-} --model /model --served-model-name glm-5.3 --tensor-parallel-size 4 --pipeline-parallel-size 2 --decode-context-parallel-size 1 \
  --quantization modelopt_fp4 --kv-cache-dtype nvfp4 --trust-remote-code --gpu-memory-utilization 0.90 \
  --reasoning-parser glm45 --tool-call-parser glm47 --enable-auto-tool-choice \
  --max-model-len 750000 --max-num-seqs 32 --max-num-batched-tokens 2048 --async-scheduling \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' --long-prefill-token-threshold 512 \
  --kv-transfer-config "$KV_JSON" --host 0.0.0.0 --port 8000 >/dev/null
for i in $(seq 1 360); do
  curl -s -m 3 -o /dev/null -w '%{http_code}' http://127.0.0.1:8000/health | grep -q 200 && { echo "[$(date -u +%T)] READY po $((i*5)) s"; exit 0; }
  st=$(docker inspect -f '{{.State.Status}}' full-glm 2>/dev/null || echo missing)
  [ "$st" != running ] && { echo "KONTENER PADL ($st)"; docker logs --tail 30 full-glm 2>&1 | cut -c1-200; exit 1; }
  sleep 5
done
echo TIMEOUT; exit 1
