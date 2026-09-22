#!/usr/bin/env bash
# E8 (2026-09-21): synthetic DMA stress on all 8 PRO GPUs of host B without vLLM (8 containers, one per GPU, dma_stress.py d2h),
# orchestrated from another host over ssh under a hang watchdog; power-cycles B first (BMC) if GPUs are still busy without a process.
# One-off kept for the record. Settings: B_SSH (required, e.g. root@testbed), SSH_OPTS (e.g. "-i ~/.ssh/key"), B_WORK (remote scratch dir,
# receives dma_stress.py and the results), LOG / EVIDENCE (local files), DUR (seconds per container), IMAGE.
set -u
B_SSH=${B_SSH:?ssh target of host B, e.g. root@testbed}
SSH_OPTS=${SSH_OPTS:-}
B_WORK=${B_WORK:-/tmp/d2}
LOG=${LOG:-/tmp/d2-e8-orchestrate.log}
EVIDENCE=${EVIDENCE:-/tmp/d2-e8-hang-evidence.txt}
IMAGE=${IMAGE:-glm53-stack:2026.09.19-rc2}
DUR=${DUR:-300}
HERE=$(cd "$(dirname "$(readlink -f "$0")")" && pwd)
SSH="ssh $SSH_OPTS -o BatchMode=yes -o ConnectTimeout=8 $B_SSH"
log() { echo "[$(date -u +%T)] $*" | tee -a "$LOG"; }
stuck=$($SSH 'nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits | awk -F", " "\$1>50 && \$2<100 {n++} END {print n+0}"' 2>/dev/null || echo ssh-fail)
log "E8 start; stuck GPUs (busy, no memory): $stuck"
if [ "$stuck" != 0 ]; then
  log "power cycle B"; $SSH 'sync; sync; ipmitool chassis power cycle' >/dev/null 2>&1; sleep 110
  for i in $(seq 1 40); do $SSH 'true' 2>/dev/null && break; sleep 10; done; sleep 45
fi
log "B up: $($SSH 'uptime | cut -c1-30; nvidia-smi -L | wc -l' 2>&1 | tr "\n" " ")"
$SSH "mkdir -p $B_WORK/results/exp_e8_dma"
scp -q $SSH_OPTS "$HERE/dma_stress.py" "$B_SSH:$B_WORK/dma_stress.py"
$SSH "docker rm -f \$(docker ps -aq -f name=dma_) >/dev/null 2>&1; for i in 0 1 2 3 4 5 7 8; do docker run -d --name dma_\$i --gpus \"\\\"device=\$i\\\"\" --ipc host --ulimit memlock=-1:-1 -e SEED=\$i -v $B_WORK:/w --entrypoint python3 $IMAGE /w/dma_stress.py $DUR 512 d2h >/dev/null; done; sleep 2; docker ps --format '{{.Names}}' | grep -c dma_"
log "8 DMA stress containers started (d2h, ${DUR}s)"
T0=$(date +%s); RES=PASS
while [ $(( $(date +%s) - T0 )) -lt $(( DUR + 120 )) ]; do
  sleep 20
  running=$($SSH 'docker ps --format "{{.Names}}" | grep -c dma_' 2>/dev/null || echo ssh-fail)
  [ "$running" = ssh-fail ] && { log "ssh failed (host down?)"; RES=HOSTDOWN; break; }
  stalled=$($SSH 'n=0; for c in $(docker ps --format "{{.Names}}" | grep dma_); do docker logs --since 40s $c 2>&1 | grep -q "it=" || n=$((n+1)); done; echo $n' 2>/dev/null)
  xid=$($SSH 'dmesg -T | grep -iE "xid|fallen" | wc -l' 2>/dev/null)
  log "running=$running stalled(no output 40s)=$stalled xid=$xid"
  if [ "$running" = 0 ]; then break; fi
  if [ "${stalled:-0}" -gt 0 ] && [ $(( $(date +%s) - T0 )) -gt 60 ]; then
    log "STALL detected in $stalled container(s); collecting"; RES=HANG
    $SSH 'for c in $(docker ps --format "{{.Names}}" | grep dma_); do echo "== $c"; docker logs --tail 2 $c 2>&1; done; nvidia-smi --query-gpu=index,utilization.gpu,power.draw,memory.used --format=csv,noheader; for d in 01 11 61 71 81 91 e1 f1; do printf "%s: " $d; lspci -s $d:00.0 -vv | grep -E "DevSta:|CESta:" | sed "s/^\s*//" | tr "\n" " "; echo; done' > "$EVIDENCE" 2>&1
    # NOTE (2026-09-22): SIGKILL of busy GPU containers leaves GPUs wedged on this driver (RESULTS §12c); prefer `docker stop -t 60`
    $SSH 'docker stop -t 60 $(docker ps -q -f name=dma_) >/dev/null 2>&1; sleep 5; nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader' >> "$EVIDENCE" 2>&1
    break
  fi
  [ "${xid:-0}" != 0 ] && { log "XID seen"; RES=XID; break; }
done
$SSH 'for c in $(docker ps -a --format "{{.Names}}" | grep dma_); do echo "== $c: $(docker logs --tail 1 $c 2>&1 | cut -c1-120)"; done' | tee -a "$LOG"
$SSH "docker logs dma_0 > $B_WORK/results/exp_e8_dma/dma_0.log 2>&1; docker logs dma_4 > $B_WORK/results/exp_e8_dma/dma_4.log 2>&1; docker rm -f \$(docker ps -aq -f name=dma_) >/dev/null 2>&1" 2>/dev/null
log "E8 RESULT: $RES"
