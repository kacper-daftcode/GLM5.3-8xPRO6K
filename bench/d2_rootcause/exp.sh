#!/usr/bin/env bash
# D2 experiment driver (host B). One experiment = start full-glm with the given knobs, run the reproducer
# (200k needle cold -> GPU hit -> CHURN x 550k -> warm -> GPU hit) under a hang watchdog, collect evidence, stop the container.
# Usage: exp.sh <label> [KV_OFFLOAD_GB=N] [NCCL_MODE=shm|p2p_sys] [EXTRA_ENV="A=1 B=2"] [KV_OFFLOAD_OPTS=json] [CHURN=4] [EXTRA_ARGS="--kernel-config ..."]
# Exit: 0 = reproducer completed (see "D2 verdict" in the RESULT line), 2 = HANG detected (container stopped/killed), 3 = container died, 4 = start failed.
#
# Hang criteria (revised 2026-09-22). The engine's status logger ("Engine 000: ...") is SILENT during a single long chunked prefill:
# a 550k prompt at 300 W takes ~150 s and prints only 1-2 lines ~6/16 s after the request starts, then nothing until the request ends.
# Log silence alone is therefore NOT a hang signal - the 60 s watchdog used on 2026-09-21 fired inside every 550k prefill (E1-E33 were
# all false positives, RESULTS §12c). A real hang shows on the hardware: GPUs spinning in NCCL (100 % util at ~130 W) or idle while a
# request is in flight, whereas a live prefill pins every PRO GPU at the 300 W cap. Criteria:
#   1. spin   : no "Engine 000" line for >= 45 s AND for >= SPIN_SECS (60) every PRO GPU draws < SPIN_W (200 W) while >= 1 shows util >= 50 %
#   2. silence: no "Engine 000" line for >= HANG_SECS (600) regardless of power (safety net; a 650k cold prefill is ~190 s)
# GPU util/power/memory is traced at 1 Hz to $OUT/gpu_trace.csv (stall analysis: gpu_trace_stalls.py).
# On HANG: evidence (py-spy, PCIe, nvidia-smi, container log), then `docker stop -t 60` BEFORE `docker kill` - SIGKILL during live multi-GPU
# work leaves GPUs at 100 % with no process (power cycle needed); a wedged engine will not stop gracefully either, but a false positive survives.
set -u
LABEL=${1:?label}; shift
for kv in "$@"; do export "$kv"; done
BENCH=$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)   # bench/ of the checkout (works through a symlink to this script)
[ -f "$BENCH/env" ] && source "$BENCH/env"                  # host settings (RESULTS_DIR, CACHE_ROOT, GPUS), see bench/env.example
RESULTS_DIR=${RESULTS_DIR:-$BENCH/../results}
CACHE_ROOT=${CACHE_ROOT:-/var/cache/glm53-stack}
CACHE=${CACHE:-$CACHE_ROOT/vllm-fi618-b1s2-p41-mtp3}         # compile cache of the candidate config; start_nsys.sh and the core-dump pipes write there
export KV_OFFLOAD_GB=${KV_OFFLOAD_GB:-0} NCCL_MODE=${NCCL_MODE:-p2p_sys} EXTRA_ENV=${EXTRA_ENV:-} KV_OFFLOAD_OPTS=${KV_OFFLOAD_OPTS:-} EXTRA_ARGS=${EXTRA_ARGS:-}
CHURN=${CHURN:-4}
GPUS=${GPUS:-0,1,2,3,4,5,7,8}          # the 8 PRO 6000 (same default as serve_full.sh; GPU 6 = the team's 5090)
HANG_SECS=${HANG_SECS:-600}; SPIN_SECS=${SPIN_SECS:-60}; SPIN_W=${SPIN_W:-200}
START_CMD=${START_CMD:-"IMAGE=glm53-stack:2026.09.19-rc2 ./run_cand.sh"}
REPRO_ARGS=${REPRO_ARGS:-"--len 200000 --churn $CHURN --churn-len 550000"}
OUT=$RESULTS_DIR/d2/exp_$LABEL; mkdir -p "$OUT"
log() { echo "[$(date -u +%T)] $*" | tee -a "$OUT/exp.log"; }
pcie_status() {  # $1 = tag
  { for d in 01 11 61 71 81 91 e1 f1; do printf "%s: " $d; lspci -s $d:00.0 -vv 2>/dev/null | grep -E "DevSta:|CESta:" | sed "s/^\s*//" | tr "\n" " "; echo; done
    for rp in 00:01.1 80:01.1; do printf "RP %s: " $rp; lspci -s $rp -vv 2>/dev/null | grep -E "DevSta:|CESta:" | sed "s/^\s*//" | tr "\n" " "; echo; done
    for i in ${GPUS//,/ }; do printf "nvidia-smi GPU%s: " $i; nvidia-smi -i $i -q 2>/dev/null | grep -iE "Replays Since Reset|Replay Number Rollovers" | sed "s/^\s*//" | tr "\n" " "; echo; done
  } > "$OUT/pcie_$1.txt" 2>&1
}
gpu_state() {  # prints "<max util> <max power W> <min power W>" over the PRO GPUs
  nvidia-smi -i "$GPUS" --query-gpu=utilization.gpu,power.draw --format=csv,noheader,nounits 2>/dev/null |
    awk -F', ' 'BEGIN{u=0;p=0;m=9999} {if($1+0>u)u=$1+0; if($2+0>p)p=$2+0; if($2+0<m)m=$2+0} END{printf "%d %d %d", u, p, m}'
}
log "=== $LABEL: KV_OFFLOAD_GB=$KV_OFFLOAD_GB NCCL_MODE=$NCCL_MODE EXTRA_ENV='$EXTRA_ENV' KV_OFFLOAD_OPTS='$KV_OFFLOAD_OPTS' EXTRA_ARGS='$EXTRA_ARGS' CHURN=$CHURN REPRO_ARGS='$REPRO_ARGS' watchdog: spin ${SPIN_SECS}s@<${SPIN_W}W, silence ${HANG_SECS}s"
dmesg -T | grep -iE "xid|fallen|Link Down" | grep -v SATA > "$OUT/dmesg_before.txt"
pcie_status before
cd "$BENCH"
if ! eval "$START_CMD" > "$OUT/start.log" 2>&1; then log "START FAILED"; tail -5 "$OUT/start.log" | tee -a "$OUT/exp.log"; exit 4; fi
log "started: $(grep -E "READY|host RAM" "$OUT/start.log" | tr "\n" " ")"
nvidia-smi -i "$GPUS" --query-gpu=timestamp,index,utilization.gpu,power.draw,memory.used --format=csv,noheader,nounits -lms 1000 > "$OUT/gpu_trace.csv" 2>/dev/null &
TRACE_PID=$!
setsid nohup python3 "$BENCH/kv_offload_reload.py" $REPRO_ARGS --out "$OUT/reload.json" > "$OUT/reload.log" 2>&1 < /dev/null &
TPID=$!
T0=$(date +%s); LAST_SEEN=$T0; SPIN_SINCE=""; RC=0
while kill -0 $TPID 2>/dev/null; do
  sleep 5
  now=$(date +%s)
  st=$(docker inspect -f '{{.State.Status}}' full-glm 2>/dev/null || echo missing)
  if [ "$st" != running ]; then log "CONTAINER $st"; RC=3; break; fi
  recent=$(docker logs --since 45s full-glm 2>&1 | grep -c "Engine 000")
  [ "$recent" != 0 ] && LAST_SEEN=$now
  silence=$(( now - LAST_SEEN ))
  read -r maxutil maxpow minpow <<< "$(gpu_state)"
  if [ $silence -ge 45 ] && [ $(( now - T0 )) -gt 60 ]; then
    if [ "$maxutil" -ge 50 ] && [ "$maxpow" -lt "$SPIN_W" ]; then [ -z "$SPIN_SINCE" ] && SPIN_SINCE=$now; else SPIN_SINCE=""; fi
    spin=0; [ -n "$SPIN_SINCE" ] && spin=$(( now - SPIN_SINCE ))
    if [ $silence -ge "$HANG_SECS" ] || [ $spin -ge "$SPIN_SECS" ]; then
      running=$(docker logs --since 900s full-glm 2>&1 | grep -E "Engine 000" | tail -1 | grep -oE "Running: [0-9]+" | grep -oE "[0-9]+")
      log "HANG detected ($([ $spin -ge "$SPIN_SECS" ] && echo "spin ${spin}s: max util ${maxutil}%, power ${minpow}-${maxpow} W" || echo "silence ${silence}s"), last Running=${running:-?}); collecting evidence"
      nvidia-smi --query-gpu=index,utilization.gpu,power.draw,memory.used --format=csv,noheader > "$OUT/nvidia-smi_hang.txt"
      for p in $(pgrep -f "Worker_PP|EngineCore"); do n=$(tr "\0" " " < /proc/$p/cmdline | grep -oE "Worker_PP[0-9]_TP[0-9]|EngineCore" | head -1); [ -n "$n" ] && { timeout 10 py-spy dump --pid $p --nonblocking > "$OUT/pyspy_$n.txt" 2>&1; timeout 30 py-spy dump --pid $p --native --nonblocking > "$OUT/pyspy_native_$n.txt" 2>&1; }; done
      pcie_status hang
      docker logs full-glm > "$OUT/container.log" 2>&1
      # GPU core dumps (CUDA_ENABLE_USER_TRIGGERED_COREDUMP=1 + CUDA_COREDUMP_PIPE in the container): trigger for two stage-1 and one stage-0 worker
      if [ -n "${COREDUMP:-}" ]; then
        for n in Worker_PP0_TP0 Worker_PP0_TP1 Worker_PP0_TP2 Worker_PP0_TP3 Worker_PP1_TP0 Worker_PP1_TP1 Worker_PP1_TP2 Worker_PP1_TP3; do
          cp=$(docker exec full-glm pgrep -f "$n" | head -1)
          pipe=$CACHE/corepipe_$cp
          if [ -n "$cp" ] && [ -p "$pipe" ]; then log "triggering core dump $n (pid $cp)"; (timeout 240 sh -c "echo dump > $pipe" &) ; fi
        done
        sleep 240; ls -la $CACHE/core_*.nvcudmp 2>/dev/null | awk "{print \$5, \$9}" | tee -a "$OUT/exp.log"
      fi
      # GPU-side truth: which kernels are resident and where they spin (needs cuda-gdb in the image: rc2-dbg)
      if docker exec full-glm test -x /usr/local/cuda-13.0/bin/cuda-gdb 2>/dev/null; then
        for n in Worker_PP0_TP0 Worker_PP1_TP0 Worker_PP1_TP1; do
          cp=$(docker exec full-glm pgrep -f "$n" | head -1)
          [ -n "$cp" ] && timeout 300 docker exec full-glm /usr/local/cuda-13.0/bin/cuda-gdb -p "$cp" -batch -ex "set pagination off" -ex "info cuda devices" -ex "info cuda kernels" -ex "info cuda blocks" -ex "cuda kernel 0 block 0 thread 0" -ex "bt" -ex "info cuda warps" -ex "cuda kernel 1 block 0 thread 0" -ex "bt" -ex "detach" > "$OUT/cudagdb_$n.txt" 2>&1
          log "cuda-gdb $n: $(grep -cE '^\*?\s+[0-9]+ ' "$OUT/cudagdb_$n.txt" 2>/dev/null) lines, kernels: $(grep -A20 'info cuda kernels' "$OUT/cudagdb_$n.txt" 2>/dev/null | grep -oE '[A-Za-z_0-9]+\(' | head -6 | tr '\n' ' ')"
        done
      fi
      if [ -n "${NSYS:-}" ]; then
        log "NSYS: nsys stop --session=hang (waits for the report)"; timeout 600 docker exec full-glm /opt/nvidia/nsight-systems/2026.3.2/bin/nsys stop --session=hang 2>&1 | tail -3 | tee -a "$OUT/exp.log"
        for i in $(seq 1 30); do [ -f $CACHE/nsys_hang.nsys-rep ] && break; sleep 5; done
        ls -la $CACHE/nsys_hang.nsys-rep 2>/dev/null | awk "{print \$5, \$9}" | tee -a "$OUT/exp.log"
        cp $CACHE/nsys_hang.nsys-rep "$OUT/" 2>/dev/null
      fi
      kill $TPID 2>/dev/null
      log "stopping container gracefully (docker stop -t 60) before kill"
      timeout 90 docker stop -t 60 full-glm >/dev/null 2>&1 || docker kill full-glm >/dev/null 2>&1
      sleep 5
      nvidia-smi --query-gpu=index,utilization.gpu,power.draw,memory.used --format=csv,noheader > "$OUT/nvidia-smi_after_kill.txt"
      RC=2; break
    fi
  else
    SPIN_SINCE=""
  fi
done
kill $TRACE_PID 2>/dev/null
VERDICT=""
if [ $RC = 0 ]; then
  wait $TPID
  VERDICT=$(grep -oE "VERDICT: .*" "$OUT/reload.log" | tail -1 | cut -c1-200)
  log "reproducer finished: $(grep -E "cold  |churn [0-9]|warm  |gpuhit " "$OUT/reload.log" | cut -c1-110 | tr "\n" "|")"
  docker logs full-glm > "$OUT/container.log" 2>&1
fi
python3 "$BENCH/d2_rootcause/gpu_trace_stalls.py" "$OUT/gpu_trace.csv" > "$OUT/gpu_trace_stalls.txt" 2>&1 && log "gpu trace: $(tail -1 "$OUT/gpu_trace_stalls.txt")"
pcie_status after
dmesg -T | grep -iE "xid|fallen|Link Down" | grep -v SATA > "$OUT/dmesg_after.txt"
log "xid lines before/after: $(wc -l < "$OUT/dmesg_before.txt")/$(wc -l < "$OUT/dmesg_after.txt"); GPUs after: $(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr "\n" ",")"
[ $RC = 0 ] && [ -z "${KEEP:-}" ] && docker stop -t 30 full-glm >/dev/null 2>&1
log "RESULT $LABEL: $([ $RC = 0 ] && echo "PASS (reproducer completed; D2 ${VERDICT:-no verdict})" || echo FAIL rc=$RC)"
exit $RC
