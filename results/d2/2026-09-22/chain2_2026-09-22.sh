#!/usr/bin/env bash
# 2026-09-22 chain 2 (after chain 1 left the baseline running): connector under concurrency.
# fairness (131k prefill + 4 decoders of 1000 tokens + new request) and smoke --conc 4, baseline vs KV_OFFLOAD_GB=334; baseline left running at the end.
set -u
R=<host-B>/a2/results/d2; L=$R/chain2_2026-09-22.log
log() { echo "[$(date -u +%T)] $*" | tee -a $L; }
while pgrep -f chain_2026-09-22.sh >/dev/null; do sleep 15; done
log "chain 1 finished; baseline running: $(docker ps --format "{{.Names}} {{.Status}}" | grep full-glm)"
cd /root/bench
log "step 1: fairness + smoke on baseline"; python3 bench_fairness.py --seed 901 --others 4 --short-tokens 1000 > $R/fairness_baseline_300w_2026-09-22.log 2>&1; log "fairness baseline rc=$?"; python3 /root/bench/smoke.py glm-5.3 --conc 4 > $R/smoke_baseline_2026-09-22.log 2>&1; log "smoke baseline rc=$?"
docker stop -t 30 full-glm >/dev/null 2>&1; sleep 5
log "step 2: start with offload 334"; KV_OFFLOAD_GB=334 IMAGE=glm53-stack:2026.09.19-rc2 ./run_cand.sh > $R/start_offload334_chain2.log 2>&1 || { log "START FAILED"; exit 1; }
log "step 3: fairness + smoke with offload"; python3 bench_fairness.py --seed 902 --others 4 --short-tokens 1000 > $R/fairness_offload334_300w_2026-09-22.log 2>&1; log "fairness offload rc=$?"; python3 /root/bench/smoke.py glm-5.3 --conc 4 > $R/smoke_offload334_2026-09-22.log 2>&1; log "smoke offload rc=$?"
log "metrics after: $(curl -s http://127.0.0.1:8000/metrics | grep -E "^vllm:kv_offload_(store|load)_bytes|^vllm:num_preemptions" | tr "\n" " " | cut -c1-300)"
docker stop -t 30 full-glm >/dev/null 2>&1; sleep 5
log "step 4: restore baseline"; IMAGE=glm53-stack:2026.09.19-rc2 ./run_cand.sh > $R/start_baseline_final_2026-09-22.log 2>&1; log "baseline rc=$? $(docker ps --format "{{.Names}} {{.Status}}" | grep full-glm)"
log "chain 2 done"
