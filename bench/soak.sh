#!/usr/bin/env bash
# Soak kandydata na B: segmenty mieszanego ruchu (soak.py) rozdzielone restartami (docker stop -t 60 + run_cand.sh),
# po kazdym segmencie parity.py (needle 32k/131k, math, tool-call) vs referencja kandydata.
# Uzycie: TAG=soak ./soak.sh 80 80 80   (minuty per segment; restart miedzy segmentami)
set -uo pipefail
cd "$(dirname "$0")"
[ -f ./env ] && source ./env                       # host settings (RESULTS_DIR, ...), see env.example
R=${R:-${RESULTS_DIR:-$(cd .. && pwd)/results}}; mkdir -p "$R"
TAG=${TAG:-soak}
REF=${REF:-$R/b1s2_parity.json}
n=$#; i=0
for MIN in "$@"; do
  i=$((i+1))
  echo "[$(date -u +%T)] segment $i/$n: $MIN min"
  python3 ./soak.py --minutes "$MIN" --log "$R/${TAG}_seg$i.jsonl" > "$R/${TAG}_seg$i.summary" 2>&1; rc=$?
  echo "[$(date -u +%T)] segment $i rc=$rc"; grep -E '"stats"' -A 30 "$R/${TAG}_seg$i.summary" | head -40
  echo "[$(date -u +%T)] parity po segmencie $i"
  python3 ./parity.py --out "$R/${TAG}_parity$i.json" --compare "$REF" --long 32768,131072 > "$R/${TAG}_parity$i.log" 2>&1; echo "parity rc=$?"
  grep -E "needle=" "$R/${TAG}_parity$i.log" | cut -c1-120
  dmesg | grep -iE "xid|nvrm" | tail -2
  if [ "$i" -lt "$n" ]; then
    echo "[$(date -u +%T)] restart $i: docker stop -t 60"
    t0=$(date +%s); docker stop -t 60 full-glm; t1=$(date +%s); echo "stop took $((t1-t0)) s"
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | paste -sd' '
    ./run_cand.sh; echo "start rc=$?"
    docker logs full-glm 2>&1 | grep -E "GPU KV cache size" | cut -c1-120
  fi
done
echo "[$(date -u +%T)] SOAK DONE"
