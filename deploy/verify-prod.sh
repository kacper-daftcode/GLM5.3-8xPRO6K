#!/usr/bin/env bash
# Weryfikacja prod po (re)starcie glm53-nvfp4-prod: health, log startowy (kwantyzacja online, KV), smoke, parity (needle/math/tool), metryki.
# Uzycie: ./verify-prod.sh [tag]   (wyniki: /tmp/verify-<tag>.*)
set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")"   # works through a symlink; smoke.py/parity.py live in ../bench
BENCH=../bench
TAG=${1:-$(date +%H%M)}
echo "[$(date -u +%T)] health: $(curl -s -m 5 -o /dev/null -w '%{http_code}' http://127.0.0.1:8000/health)  models: $(curl -s -m 5 http://127.0.0.1:8000/v1/models | python3 -c 'import sys,json; print([m["id"] for m in json.load(sys.stdin)["data"]])' 2>/dev/null)"
echo "--- log startowy:"
sudo docker logs glm53-nvfp4-prod 2>&1 | grep -E "TQ_B1: online NVFP4 for|TQ_FP8: (lm_head|eh_proj)|Selected .*Kernel|GPU KV cache size|Maximum concurrency|Loading weights took|torch.compile took|ERROR|Traceback" | cut -c1-200 | sort -u | head -20
echo "--- dmesg (ostatnie NVRM/Xid):"; sudo dmesg -T | grep -iE "xid|nvrm" | tail -3
echo "--- smoke:"
timeout 900 python3 "$BENCH/smoke.py" glm-5.3 --conc 4 2>&1 | grep -E "===|SINGLE|tok/s|finish=" | cut -c1-160 | tee /tmp/verify-$TAG.smoke
echo "--- parity (needle 32k/131k, math, tool):"
timeout 1200 python3 "$BENCH/parity.py" --out /tmp/verify-$TAG.parity.json --long 32768,131072 2>&1 | grep -E "needle=|zapisano" | cut -c1-140
echo "--- metryki:"
curl -s http://127.0.0.1:8000/metrics | grep -E "^vllm:(num_requests_running|num_requests_waiting|spec_decode_num_drafts_total|spec_decode_num_accepted_tokens_total|request_success_total)" | sed 's/{[^}]*}//' | awk '{a[$1]+=$2} END{for(k in a) print "  "k, a[k]}'
echo "[$(date -u +%T)] verify done"
