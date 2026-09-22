#!/usr/bin/env bash
# Testbed launcher for the production candidate configuration (mirrors deploy/serve-glm53-prod.sh CAND=1):
# TP4xPP2 DCP1 MTP3 async, partition 41/37, online FP8 per-channel for BF16 linears (+ draft eh_proj), MTP draft experts in online NVFP4,
# long-prefill threshold 512 (shared-only).
# Knobs: IMAGE PART MTP CACHE EXTRA_ENV, and (same names/semantics as the prod script) FP8 (=FP8_LINEARS; empty = no online FP8),
# FP8_CHANNEL, NVFP4_MOE (empty = BF16 draft experts), LONG_PREFILL (0 = no threshold), KV_OFFLOAD_GB (D2: native CPU KV tier, 0 = off).
# D2: KV_OFFLOAD_GB=400 IMAGE=glm53-stack:2026.09.19-rc2 ./run_cand.sh     (400 GiB pinned host RAM = ~2x the GPU KV pool)
# Ablations: FP8=fused_qkv_a_proj,q_b_proj,o_proj,shared_experts,indexer.wq_b,eh_proj ./run_cand.sh   (no lm_head)
#            FP8= NVFP4_MOE= ./run_cand.sh                                                            (no online quantization at all)
# The compile cache must be separate per (image, FP8 set, partition, MTP): a non-default quant set gets an automatic "-q<hash>" suffix.
set -euo pipefail
[ -f "$(dirname "$0")/env" ] && source "$(dirname "$0")/env"   # host settings (MODEL, CACHE_ROOT, ...), see env.example
CACHE_ROOT=${CACHE_ROOT:-/var/cache/glm53-stack}
IMAGE=${IMAGE:-vllm-nightly-fi614:nvfp4-fi618}
PART=${PART:-41,37}
MTP=${MTP:-3}
FP8_DEFAULT=fused_qkv_a_proj,q_b_proj,o_proj,shared_experts,indexer.wq_b,lm_head,eh_proj
FP8=${FP8-$FP8_DEFAULT}                       # FP8= (empty) disables online FP8
FP8_CHANNEL=${FP8_CHANNEL:-1}
NVFP4_MOE=${NVFP4_MOE-layers.78.mlp.experts}  # NVFP4_MOE= (empty) keeps the draft experts in BF16
LONG_PREFILL=${LONG_PREFILL:-512}
LONG_PREFILL_DYNAMIC=${LONG_PREFILL_DYNAMIC:-}   # D3: "N:T" (image >= 2026.09.19-rc2), empty = static threshold
KV_OFFLOAD_GB=${KV_OFFLOAD_GB:-0}      # D2: native CPU KV tier (vLLM OffloadingConnector); GiB of pinned host RAM summed over all 8 workers, 0 = off.
                                       # Must exceed the aggregate GPU KV pool (~200 GiB for 1.81M tokens) to add any hit rate.
                                       # CAUTION: torch pins one tensor per layer and rounds each up to a power of two, so the real footprint is
                                       # 1.1-2x the request depending on where the per-layer tensor size lands. Sweet spots for rc2 TP4xPP2 41/37
                                       # (block 64, 352 B/tok MLA + 8448 B/block indexer): 334 -> ~390 GB real (2.8M tok), 668 -> ~775 GB real (5.6M tok).
                                       # 400 on B (755 GB RAM) pinned ~700 GB and exhausted swap (2026-09-21). The script checks MemAvailable after start.
KV_OFFLOAD_OPTS=${KV_OFFLOAD_OPTS:-}   # extra kv_connector_extra_config fields, JSON without spaces, e.g. '"block_size":256,"eviction_policy":"arc"'
KV_OFFLOAD_CONNECTOR=${KV_OFFLOAD_CONNECTOR:-OffloadingConnector}   # or SimpleCPUOffloadConnector (different scheduler/worker, same cuMemcpyBatchAsync DMA)
EXTRA_ENV=${EXTRA_ENV:-}
EXTRA_ARGS=${EXTRA_ARGS:-}   # extra vLLM CLI args appended verbatim (word-split), e.g. EXTRA_ARGS="--enforce-eager"
QSUF=""
if [ "$FP8" != "$FP8_DEFAULT" ] || [ "$FP8_CHANNEL" != 1 ] || [ "$NVFP4_MOE" != layers.78.mlp.experts ]; then
  QSUF="-q$(printf '%s|%s|%s' "$FP8" "$FP8_CHANNEL" "$NVFP4_MOE" | md5sum | cut -c1-6)"
fi
CACHE=${CACHE:-$CACHE_ROOT/vllm-fi618-b1s2-p41$([ "$MTP" = 3 ] && echo -mtp3 || true)$QSUF$([ "${PP:-2}" != 2 ] && echo "-tp${TP:-4}pp${PP:-2}" || true)}   # one compile cache per config
TP=${TP:-4}; PP=${PP:-2}; DCP=${DCP:-1}   # topology (TP*PP must be 8); PART applies only when PP>1
ENVX=""; [ "$PP" -gt 1 ] && ENVX="VLLM_PP_LAYER_PARTITION=$PART"
[ -n "$FP8" ] && ENVX="$ENVX VLLM_TQ_FP8_LINEARS=$FP8 VLLM_TQ_FP8_CHANNEL=$FP8_CHANNEL"
[ -n "$NVFP4_MOE" ] && ENVX="$ENVX VLLM_TQ_NVFP4_MOE=$NVFP4_MOE"
[ -n "$LONG_PREFILL_DYNAMIC" ] && ENVX="$ENVX VLLM_LONG_PREFILL_THRESHOLD_DYNAMIC=$LONG_PREFILL_DYNAMIC"
EXTRA=""
[ "$LONG_PREFILL" -gt 0 ] && EXTRA="--long-prefill-token-threshold $LONG_PREFILL"
if [ "$KV_OFFLOAD_GB" -gt 0 ]; then
  # serve_full.sh word-splits EXTRA, so the JSON must not contain spaces
  KV_JSON="{\"kv_connector\":\"$KV_OFFLOAD_CONNECTOR\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"cpu_bytes_to_use\":$((KV_OFFLOAD_GB*1024*1024*1024))${KV_OFFLOAD_OPTS:+,$KV_OFFLOAD_OPTS}}}"
  EXTRA="$EXTRA --kv-transfer-config $KV_JSON"
fi
[ -n "$EXTRA_ARGS" ] && EXTRA="$EXTRA $EXTRA_ARGS"
echo "[$(date -u +%T)] run_cand: image=$IMAGE part=$PART mtp=$MTP fp8=${FP8:--} ch=$FP8_CHANNEL nvfp4-moe=${NVFP4_MOE:--} long-prefill=$LONG_PREFILL kv-offload=${KV_OFFLOAD_GB}G/${KV_OFFLOAD_CONNECTOR}${KV_OFFLOAD_OPTS:+($KV_OFFLOAD_OPTS)} cache=$CACHE"
if [ "$KV_OFFLOAD_GB" -gt 0 ]; then
  avail_gb=$(awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo)
  need_gb=$((KV_OFFLOAD_GB * 5 / 4 + 100))   # lower bound (power-of-two rounding can double it); keep >=100 GB for page cache / other containers
  if [ "$avail_gb" -lt "$need_gb" ]; then echo "KV_OFFLOAD_GB=$KV_OFFLOAD_GB needs >= ${need_gb} GB free host RAM, MemAvailable=${avail_gb} GB - refusing to start"; exit 1; fi
fi
rc=0
IMAGE=$IMAGE CACHE=$CACHE TP=$TP PP=$PP DCP=$DCP MTP=$MTP ASYNC=1 ENVX="$ENVX $EXTRA_ENV" EXTRA="$EXTRA" "$(dirname "$0")/serve_full.sh" || rc=$?
if [ "$KV_OFFLOAD_GB" -gt 0 ] && [ "$rc" = 0 ]; then
  avail_gb=$(awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo)
  shm_gb=$(awk '/^Shmem:/ {printf "%d", $2/1048576}' /proc/meminfo)
  echo "[$(date -u +%T)] host RAM after start: pinned(shmem)=${shm_gb} GB MemAvailable=${avail_gb} GB$( [ "$avail_gb" -lt 100 ] && echo '  <-- WARNING: too little headroom, lower KV_OFFLOAD_GB (see comment above)')"
fi
exit $rc
