#!/usr/bin/env bash
# 2026-09-22 chain: D2 gate at 650k (334 GiB), decode bench with offload, 32 GiB pool eviction path (autotune ON), decode bench baseline.
set -u
R=<host-B>/a2/results/d2; L=$R/chain_2026-09-22.log
log() { echo "[$(date -u +%T)] $*" | tee -a $L; }
cd <host-B>/a2/d2
log "step 1: d2gate_334_650k"; KEEP=1 ./exp.sh d2gate_334_650k KV_OFFLOAD_GB=334 REPRO_ARGS="--len 650000 --churn 3 --churn-len 550000" >> $L 2>&1; log "step 1 rc=$?"
log "step 2: bench_decode with offload 334 (server kept)"; (cd /root/bench && python3 bench_decode.py --conc 1,16 --tokens 256 --rounds 3 > $R/bench_decode_offload334_300w_2026-09-22.log 2>&1); log "step 2 rc=$? $(grep -E "ms/step|conc" $R/bench_decode_offload334_300w_2026-09-22.log | tail -4 | tr "\n" " ")"
docker stop -t 30 full-glm >/dev/null 2>&1; sleep 5
log "step 3: pool32_autotune_on (eviction path)"; ./exp.sh pool32_autotune_on KV_OFFLOAD_GB=32 CHURN=4 >> $L 2>&1; log "step 3 rc=$?"
log "step 4: baseline (no offload) + bench_decode"; (cd /root/bench && IMAGE=glm53-stack:2026.09.19-rc2 ./run_cand.sh > $R/start_baseline_2026-09-22.log 2>&1 && python3 bench_decode.py --conc 1,16 --tokens 256 --rounds 3 > $R/bench_decode_baseline_300w_2026-09-22.log 2>&1); log "step 4 rc=$? $(grep -E "ms/step|conc" $R/bench_decode_baseline_300w_2026-09-22.log | tail -4 | tr "\n" " ")"
log "chain done; full-glm baseline left running: $(docker ps --format "{{.Names}} {{.Status}}" | grep full-glm)"
