#!/usr/bin/env bash
# Full measurement pass against a running server (testbed): smoke -> parity -> CE -> long-context needle -> decode 1/4/16/32 -> prefill -> fairness -> GSM8K.
# Writes results/<tag>/* and results/<tag>/summary.md. Reference files for --compare live in results/ (parity: reference_parity.json, CE: base_lp.json).
# Usage: bench/run_all.sh <tag> [--quick]      (quick = skip GSM8K and 262k prefill; ~10 min instead of ~30)
set -uo pipefail
cd "$(dirname "$0")"
TAG=${1:?tag}; QUICK=${2:-}
OUT=../results/$TAG; mkdir -p "$OUT"
URL=${URL:-http://127.0.0.1:8000}; MODEL=${MODEL:-glm-5.3}
REF_PARITY=${REF_PARITY:-../results/reference_parity.json}; REF_LP=${REF_LP:-../results/base_lp.json}
log() { echo "[$(date -u +%T)] $*" | tee -a "$OUT/run_all.log"; }
run() { local name=$1; shift; log "== $name"; "$@" 2>&1 | tee "$OUT/$name.log" | tail -${TAIL:-12}; }

log "server: $(curl -s -m 5 "$URL/v1/models" | python3 -c 'import sys,json; print([m["id"] for m in json.load(sys.stdin)["data"]])' 2>/dev/null)"
docker logs "${CONTAINER:-full-glm}" 2>&1 | grep -E "GPU KV cache size|non-default args|TQ_B1: online NVFP4 for|Selected .*Kernel" | cut -c1-240 | sort -u > "$OUT/server_config.txt" 2>/dev/null
run smoke      python3 smoke.py "$MODEL" --conc 4
run parity     python3 parity.py --out "$OUT/parity.json" $( [ -f "$REF_PARITY" ] && echo --compare "$REF_PARITY" ) --long 32768,131072
run lp_probe   python3 lp_probe.py --out "$OUT/lp.json" $( [ -f "$REF_LP" ] && echo --compare "$REF_LP" )
run needle     python3 needle_len.py --lens 262144,650000 --fmt chat --seed 9
run decode     python3 bench_decode.py --conc 1,4,16,32 --tokens 256 --rounds 3
run prefill    python3 bench_prefill.py --model "$MODEL" --lens $( [ -n "$QUICK" ] && echo 32768,131072 || echo 32768,131072,262144 ) --seed $((RANDOM % 900 + 100))
run fairness   python3 bench_fairness.py --seed $((RANDOM % 900 + 100))
[ -z "$QUICK" ] && run gsm8k python3 gsm8k_eval.py --n 500 --conc 8 --out "$OUT/gsm8k.json"

{
  echo "# $TAG — $(date -u +%F' '%T) UTC"
  echo; echo '```'; cat "$OUT/server_config.txt"; echo '```'
  echo; echo "## decode (ms/step | tok/s | acceptance)"; grep -E "^conc=" "$OUT/decode.log" | grep -v " r0:" | sed 's/  */ /g'
  echo; echo "## prefill"; grep -E "^ctx=" "$OUT/prefill.log"
  echo; echo "## fairness"; grep -E "dlugi prefill|^ (old|new):" "$OUT/fairness.log" | cut -c1-200
  echo; echo "## smoke"; grep -E "SINGLE|przepustowosc" "$OUT/smoke.log"
  echo; echo "## parity"; grep -E "needle=(True|False)|IDENTYCZNE|ROZNE" "$OUT/parity.log" | cut -c1-120
  echo; echo "## teacher-forced CE"; grep -E "^ *(pl|code|mix):.*CE" "$OUT/lp_probe.log"
  echo; echo "## long-context needle (chat)"; grep -E "needle=" "$OUT/needle.log" | cut -c1-160
  [ -f "$OUT/gsm8k.json" ] && { echo; echo "## GSM8K"; python3 -c "import json; print(json.load(open('$OUT/gsm8k.json'))['summary'])"; }
} > "$OUT/summary.md"
log "summary: $OUT/summary.md"
