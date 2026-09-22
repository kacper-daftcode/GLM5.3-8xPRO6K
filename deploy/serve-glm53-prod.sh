#!/usr/bin/env bash
# GLM-5.3 NVFP4 (incoai) jako PROD pod ruch wspolbiezny. Alias glm-5.2 zeby dotychczasowi klienci nie musieli nic zmieniac.
# restart=unless-stopped, GPU po UUID (common.sh), trwaly cache torch.compile/Triton.
# Ustawienia hosta (MODEL_DIR, CACHE_ROOT) z pliku "env" obok skryptu (git-ignored, wzor env.example) albo ze srodowiska.
#
# Dwie konfiguracje:
#  * stara (domyslna, rollback):  NCCL_MODE=p2p_sys MTP=5 DCP=2 ./serve-glm53-prod.sh
#      obraz nvfp4, TP8 DCP2 MTP5, cache $CACHE_ROOT/vllm
#  * kandydat 2026-09-18 (soak na B; od 19.09 MTP3): CAND=1 ./serve-glm53-prod.sh
#      obraz nvfp4-fi618 (FlashInfer 0.6.18 + patche: PP+MTP, fairness, online FP8/NVFP4), TP4xPP2 DCP1 MTP3, partition 41/37,
#      online FP8 per-channel dla warstw BF16 (+eh_proj drafta), eksperci drafta MTP w NVFP4, prog prefillu 512 (shared-only),
#      osobny cache $CACHE_ROOT/vllm-fi618-cand-mtp3 (MTP5: CAND=1 MTP=5 CACHE=$CACHE_ROOT/vllm-fi618-cand). Kazde pokretlo mozna nadpisac osobno.
#      Od 20.09 obraz glm53-stack:2026.09.19-rc2 + D3 (LONG_PREFILL_DYNAMIC=4:128); od 22.09 13:06 UTC + D2: tier KV w RAM KV_OFFLOAD_GB=668
#      (720 GiB pinned, start ~4 min zamiast 2.5 przez pinning; KV_OFFLOAD_GB=0 = bez tieru, ten sam obraz i cache).
set -euo pipefail
source "$(dirname "$(readlink -f "$0")")/common.sh"   # works through a symlink
MODEL_DIR=${MODEL_DIR:?set MODEL_DIR (ModelOpt NVFP4 checkpoint directory) in deploy/env or the environment}
CAND=${CAND:-0}
if [ "$CAND" = 1 ]; then
  IMAGE=${IMAGE:-glm53-stack:2026.09.19-rc2}   # od 20.09 16:00 UTC prod = rc2 (rc1 + dynamiczny chunk D3); poprzedni obraz: vllm-nightly-fi614:nvfp4-fi618
  TP=${TP:-4}; PP=${PP:-2}; DCP=${DCP:-1}; MTP=${MTP:-3}   # MTP3 >= MTP5 na tym stacku (D1, 18.09): single rowne, @4-32 +4-7%, KV +1%
  NCCL_MODE=${NCCL_MODE:-p2p_sys}
  PP_PARTITION=${PP_PARTITION:-41,37}
  # "${VAR-default}" (not ":-"): an explicitly empty FP8_LINEARS= / NVFP4_MOE= must stay empty (= online quantization off), see the variants above
  FP8_LINEARS=${FP8_LINEARS-fused_qkv_a_proj,q_b_proj,o_proj,shared_experts,indexer.wq_b,lm_head,eh_proj}
  FP8_CHANNEL=${FP8_CHANNEL:-1}
  NVFP4_MOE=${NVFP4_MOE-layers.78.mlp.experts}
  LONG_PREFILL=${LONG_PREFILL:-512}
  LONG_PREFILL_DYNAMIC=${LONG_PREFILL_DYNAMIC-4:128}   # D3: chunk 128 gdy >4 requestow; LONG_PREFILL_DYNAMIC= (puste) = staly prog 512 jak do 20.09
  KV_OFFLOAD_GB=${KV_OFFLOAD_GB:-668}   # D2 (prod od 22.09 13:06 UTC): tier KV w RAM 668 GiB = 720 GiB pinned (5.6M tokenow, 3.1x pula GPU); KV_OFFLOAD_GB=0 wylacza (rollback bez zmiany obrazu/cache)
  CACHE=${CACHE:-$CACHE_ROOT/vllm-fi618-cand-mtp3}
fi
IMAGE=${IMAGE:-vllm-nightly-fi614:nvfp4}   # nvfp4-car = wymuszony custom all-reduce po PCIe; nvfp4-fi618 = FlashInfer 0.6.18 + patche
CAR_MAX=${CAR_MAX:-2097152}               # prog bajtow dla custom AR na PCIe (0 = wylaczony; obraz fi618 ma domyslnie 0)
MAX_SEQS=${MAX_SEQS:-32}
MTP=${MTP:-3}                     # 0 = bez spekulacji; 3/5 = MTP (bezstratne, ~2.5x szybszy decode przy 1-4 strumieniach)
BATCHED=${BATCHED:-2048}          # 2048: krok z dlugim prefillem ~1.2 s zamiast ~5 s -> decode innych nie staje
PARTIAL=${PARTIAL:-0}             # 1 => --max-num-partial-prefills (UWAGA: odrzucane przez ta wersje przy MTP/DCP - crash-loop)
PARTIAL_ARGS=(); [ "$PARTIAL" = 1 ] && PARTIAL_ARGS=(--long-prefill-token-threshold 4096 --max-num-partial-prefills 8 --max-long-partial-prefills 1)
LONG_PREFILL=${LONG_PREFILL:-0}   # N>0 => --long-prefill-token-threshold N (chunk dlugiego prefillu; w fi618 tylko gdy >1 request)
[ "$LONG_PREFILL" -gt 0 ] && PARTIAL_ARGS+=(--long-prefill-token-threshold "$LONG_PREFILL")
LONG_PREFILL_DYNAMIC=${LONG_PREFILL_DYNAMIC:-}  # D3 (obraz >= 2026.09.19-rc2): "N:T" => powyzej N requestow chunk = T (np. 4:128); puste = staly prog
KV_OFFLOAD_GB=${KV_OFFLOAD_GB:-0}  # D2: natywny tier KV w RAM (vLLM OffloadingConnector); GiB przypietej pamieci hosta lacznie dla 8 workerow, 0 = wylaczony.
                                  # Musi przekraczac laczna pule KV na GPU (~200 GiB przy 1.81M tok), inaczej tylko dubluje GPU.
                                  # UWAGA: torch przypina osobny tensor per warstwa i zaokragla kazdy W GORE do potegi dwojki -> realne zuzycie RAM to 1.1-2x
                                  # zadanej wartosci. Zmierzone punkty dla rc2 TP4xPP2 41/37: 334 -> 360 GiB pinned (2.8M tok, B 21-22.09), 668 -> 720 GiB (5.6M tok, A 22.09).
                                  # Wartosci miedzy 668 a 1336 zaokraglaja sie do ~1440 GiB (!), nie uzywac.
                                  # Inne wartosci moga zajac 2x wiecej (400 na B = ~700 GB i pelny swap, 21.09). Po starcie skrypt wypisuje MemAvailable.
KV_OFFLOAD_OPTS=${KV_OFFLOAD_OPTS:-}  # dodatkowe pola kv_connector_extra_config (JSON bez nawiasow zewnetrznych), np. '"block_size":256,"eviction_policy":"arc"'
KV_ARGS=()
if [ "$KV_OFFLOAD_GB" -gt 0 ]; then
  KV_ARGS=(--kv-transfer-config "{\"kv_connector\":\"OffloadingConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"cpu_bytes_to_use\":$((KV_OFFLOAD_GB*1024*1024*1024))${KV_OFFLOAD_OPTS:+,$KV_OFFLOAD_OPTS}}}")
  avail_gb=$(awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo)
  need_gb=$((KV_OFFLOAD_GB * 5 / 4 + 100))   # dolne oszacowanie (zaokraglanie moze podwoic); >=100 GB zostawiamy na page cache wag i inne kontenery
  if [ "$avail_gb" -lt "$need_gb" ] && [ "${DRY:-0}" != 1 ]; then echo "KV_OFFLOAD_GB=$KV_OFFLOAD_GB wymaga >= ${need_gb} GB wolnego RAM, MemAvailable=${avail_gb} GB - stop"; exit 1; fi
fi
TP=${TP:-8}                       # tensor parallel
PP=${PP:-1}                       # pipeline parallel (TP*PP musi byc = 8); PP=2 => stage 0 = NUMA0 (01/11/61/71), stage 1 = NUMA1 (81/91/E1/F1)
DCP=${DCP:-2}                     # decode context parallel (dzieli TP); pula KV ~ DCP x 0.8M tok
PP_PARTITION=${PP_PARTITION:-}    # np. 41,37 => -e VLLM_PP_LAYER_PARTITION (balans pamieci miedzy stage'ami; patrz HANDOFF)
FP8_LINEARS=${FP8_LINEARS:-}      # online FP8 W8A8 dla warstw BF16 (obraz fi618): podciagi prefixow, np. fused_qkv_a_proj,q_b_proj,...
FP8_CHANNEL=${FP8_CHANNEL:-0}     # 1 => skala FP8 per wiersz wagi (dokladniej, ten sam koszt)
NVFP4_MOE=${NVFP4_MOE:-}          # online NVFP4 dla MoE BF16 (eksperci drafta MTP), np. layers.78.mlp.experts
CACHE=${CACHE:-$CACHE_ROOT/vllm} # trwaly cache torch.compile/Triton na hoscie; OSOBNY katalog per obraz/zestaw FP8/partition
NCCL_MODE=${NCCL_MODE:-shm}       # shm = NCCL_P2P_DISABLE=1 (stare) | p2p_sys = P2P on + NCCL_P2P_LEVEL=SYS (zmierzone -30% latencji all-reduce)
PROFILE=${PROFILE:-0}             # 1 => torch profiler przez /start_profile /stop_profile, slady w /root/.cache/vllm/prof
PROF_ARGS=(); [ "$PROFILE" = 1 ] && PROF_ARGS=(--profiler-config '{"profiler":"torch","torch_profiler_dir":"/root/.cache/vllm/prof","torch_profiler_with_stack":false}')
SPEC_ARGS=()
if [ "$MTP" -gt 0 ]; then SPEC_ARGS=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$MTP}"); fi
CAR_ENV=(); case "$IMAGE" in *car*) CAR_ENV=(-e VLLM_CUSTOM_AR_PCIE_MAX_SIZE="$CAR_MAX");; esac
TQ_ENV=()
[ -n "$PP_PARTITION" ] && TQ_ENV+=(-e VLLM_PP_LAYER_PARTITION="$PP_PARTITION")
[ -n "$FP8_LINEARS" ] && TQ_ENV+=(-e VLLM_TQ_FP8_LINEARS="$FP8_LINEARS" -e VLLM_TQ_FP8_CHANNEL="$FP8_CHANNEL")
[ -n "$NVFP4_MOE" ] && TQ_ENV+=(-e VLLM_TQ_NVFP4_MOE="$NVFP4_MOE")
[ -n "$LONG_PREFILL_DYNAMIC" ] && TQ_ENV+=(-e VLLM_LONG_PREFILL_THRESHOLD_DYNAMIC="$LONG_PREFILL_DYNAMIC")
if [ "$PP" -gt 1 ] && [ "$MTP" -gt 0 ] && [[ "$IMAGE" != *fi618* && "$IMAGE" != glm53-stack:* ]]; then echo "PP>1 z MTP wymaga obrazu z patch_pp_mtp (nvfp4-fi618 albo glm53-stack:*)"; exit 1; fi
DRY=${DRY:-0}                     # 1 => tylko wypisz komende docker run (bez zatrzymywania/startowania)
if [ "$DRY" = 1 ]; then sudo() { echo "+ sudo $*"; }; stop_big_gpu_containers() { echo "+ (stop_big_gpu_containers)"; }; wait_ready() { :; }; fi
sudo mkdir -p "$CACHE"
stop_big_gpu_containers
sudo docker rm -f glm53-nvfp4-prod >/dev/null 2>&1 || true
PROD_ARGS=(); prev=""
for a in "${COMMON_DOCKER_ARGS[@]}"; do
  if [ "$prev" = "--restart" ] && [ "$a" = "no" ]; then a=unless-stopped; fi
  if [ "$NCCL_MODE" = p2p_sys ] && [ "$a" = "NCCL_P2P_DISABLE=1" ]; then a=NCCL_P2P_DISABLE=0; fi
  if [ "$a" = "$CACHE_ROOT/vllm:/root/.cache/vllm" ]; then a="$CACHE:/root/.cache/vllm"; fi
  PROD_ARGS+=("$a"); prev="$a"
done
[ "$NCCL_MODE" = p2p_sys ] && PROD_ARGS+=(-e NCCL_P2P_LEVEL=SYS)
echo "[$(date -u +%T)] start glm53-nvfp4-prod z $MODEL_DIR (image=$IMAGE, NCCL=$NCCL_MODE, TP=$TP PP=$PP DCP=$DCP partition=${PP_PARTITION:--}, max-num-seqs=$MAX_SEQS, MTP=$MTP, batched=$BATCHED, long-prefill=$LONG_PREFILL dyn=${LONG_PREFILL_DYNAMIC:--}, fp8=${FP8_LINEARS:--} ch=$FP8_CHANNEL, nvfp4-moe=${NVFP4_MOE:--}, kv-offload=${KV_OFFLOAD_GB}G${KV_OFFLOAD_OPTS:+($KV_OFFLOAD_OPTS)}, cache=$CACHE, PROFILE=$PROFILE)"
sudo docker run --name glm53-nvfp4-prod "${PROD_ARGS[@]}" \
  -v "$MODEL_DIR":/model \
  "${CAR_ENV[@]}" "${TQ_ENV[@]}" "$IMAGE" \
  --model /model --served-model-name glm-5.3 glm-5.2 \
  --tensor-parallel-size "$TP" --pipeline-parallel-size "$PP" --decode-context-parallel-size "$DCP" \
  --quantization modelopt_fp4 --kv-cache-dtype nvfp4 \
  --reasoning-parser glm45 --tool-call-parser glm47 --enable-auto-tool-choice \
  --trust-remote-code --gpu-memory-utilization 0.90 \
  --max-model-len 750000 --max-num-seqs "$MAX_SEQS" --max-num-batched-tokens "$BATCHED" \
  --async-scheduling "${SPEC_ARGS[@]}" "${PROF_ARGS[@]}" "${PARTIAL_ARGS[@]}" "${KV_ARGS[@]}" \
  --host 0.0.0.0 --port "$PORT"
wait_ready 1800 glm53-nvfp4-prod
if [ "$KV_OFFLOAD_GB" -gt 0 ] && [ "$DRY" != 1 ]; then
  echo "[$(date -u +%T)] RAM hosta po starcie: pinned(shmem)=$(awk '/^Shmem:/ {printf "%d", $2/1048576}' /proc/meminfo) GB MemAvailable=$(awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo) GB (ponizej 100 GB = obnizyc KV_OFFLOAD_GB)"
fi
