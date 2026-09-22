#!/usr/bin/env bash
# E15 launcher: Qwen3-1.7B (dense, non-MLA attention) on the same image/topology TP4xPP2 with KV offload.
set -u
KV_OFFLOAD_GB=${KV_OFFLOAD_GB:-32}
KV_JSON="{\"kv_connector\":\"OffloadingConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"cpu_bytes_to_use\":$((KV_OFFLOAD_GB*1024*1024*1024))}}"
docker rm -f full-glm >/dev/null 2>&1
docker run -d --name full-glm --runtime nvidia --gpus '"device=0,1,2,3,4,5,7,8"' --network host --ipc host --shm-size 32g --ulimit memlock=-1:-1 \
  -e HF_HUB_OFFLINE=1 -e NCCL_P2P_DISABLE=0 -e NCCL_P2P_LEVEL=SYS -e VLLM_ENGINE_READY_TIMEOUT_S=3600 \
  -v /root/.cache/huggingface:/root/.cache/huggingface -v /root/.cache/vllm-e15-qwen:/root/.cache/vllm \
  glm53-stack:2026.09.19-rc2 --model Qwen/Qwen3-1.7B --served-model-name glm-5.3 \
  --tensor-parallel-size 4 --pipeline-parallel-size 2 --gpu-memory-utilization 0.85 \
  --max-model-len 32768 --max-num-seqs 32 --max-num-batched-tokens 2048 --async-scheduling \
  --kv-transfer-config "$KV_JSON" --host 0.0.0.0 --port 8000 >/dev/null
for i in $(seq 1 240); do
  curl -s -m 3 -o /dev/null -w '%{http_code}' http://127.0.0.1:8000/health | grep -q 200 && { echo "[$(date -u +%T)] READY po $((i*5)) s"; exit 0; }
  st=$(docker inspect -f '{{.State.Status}}' full-glm 2>/dev/null || echo missing)
  [ "$st" != running ] && { echo "KONTENER PADL ($st)"; docker logs --tail 25 full-glm 2>&1 | cut -c1-200; exit 1; }
  sleep 5
done
echo TIMEOUT; exit 1
