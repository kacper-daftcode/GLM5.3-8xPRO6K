#!/usr/bin/env bash
# D6: production health check for glm53-nvfp4-prod, meant to run every minute from cron. Stateless except for a small state file.
# Checks: container running, /health 200, requests waiting (>WAIT_MAX in two consecutive runs), new preemptions, new Xid/NVRM lines,
# ERROR/Traceback lines in the container log, MTP acceptance over the last ACC_WINDOW minutes (from /metrics deltas) below ACC_MIN.
# Output: one line per finding to ALERT_LOG (and stdout); ALERT_CMD (optional) receives the message as $1 (e.g. a Slack/Mattermost webhook curl).
# Config via deploy/env or environment: PORT ALERT_LOG ALERT_CMD WAIT_MAX ACC_MIN ACC_WINDOW STATE.
[ -f "$(dirname "$(readlink -f "$0")")/env" ] && source "$(dirname "$(readlink -f "$0")")/env"
PORT=${PORT:-8000}; NAME=${NAME:-glm53-nvfp4-prod}
ALERT_LOG=${ALERT_LOG:-/var/tmp/glm53-alerts.log}; ALERT_CMD=${ALERT_CMD:-}
WAIT_MAX=${WAIT_MAX:-2}; ACC_MIN=${ACC_MIN:-2.6}; ACC_WINDOW=${ACC_WINDOW:-60}; STATE=${STATE:-/var/tmp/glm53-alert.state}
K=${MTP_K:-3}
now=$(date +%s); ts=$(date -u +%FT%TZ)
declare -A st
[ -f "$STATE" ] && while IFS='=' read -r k v; do [ -n "$k" ] && st[$k]=$v; done < "$STATE"
alerts=()
alert() { alerts+=("$1"); }

# 1. container + health
running=$(sudo -n docker inspect -f '{{.State.Running}}' "$NAME" 2>/dev/null || echo false)
[ "$running" = true ] || alert "CRITICAL container $NAME not running"
code=$(curl -s -m 5 -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/health")
[ "$code" = 200 ] || alert "CRITICAL /health -> ${code:-timeout}"

# 2. metrics
m=$(curl -s -m 5 "http://127.0.0.1:$PORT/metrics" 2>/dev/null)
get() { echo "$m" | grep -E "^vllm:$1(\{|\s)" | sed 's/{[^}]*}//' | awk '{s+=$2} END{printf "%d", s}'; }
if [ -n "$m" ]; then
  waiting=$(get num_requests_waiting); running_req=$(get num_requests_running)
  pre=$(get num_preemptions_total); acc=$(get spec_decode_num_accepted_tokens_total); dr=$(get spec_decode_num_drafts_total)
  if [ "$waiting" -gt "$WAIT_MAX" ]; then
    [ "${st[waiting_high]:-0}" = 1 ] && alert "WARN requests waiting=$waiting (running=$running_req) for two consecutive checks"
    st[waiting_high]=1
  else st[waiting_high]=0; fi
  [ -n "${st[pre]:-}" ] && [ "$pre" -gt "${st[pre]}" ] && alert "WARN preemptions +$((pre - st[pre])) (total $pre)"
  st[pre]=$pre
  # acceptance over the window: compare with the snapshot taken >= ACC_WINDOW minutes ago
  if [ -n "${st[acc_t]:-}" ] && [ $((now - st[acc_t])) -ge $((ACC_WINDOW * 60)) ]; then
    dd=$((dr - st[acc_d])); da=$((acc - st[acc_a]))
    if [ "$dd" -ge 200 ]; then
      a=$(awk -v a=$da -v d=$dd -v k=$K 'BEGIN{printf "%.2f", 1 + a/d}')
      awk -v a=$a -v min=$ACC_MIN 'BEGIN{exit !(a < min)}' && alert "WARN MTP acceptance $a over the last $ACC_WINDOW min ($dd drafts) < $ACC_MIN"
    fi
    st[acc_t]=$now; st[acc_a]=$acc; st[acc_d]=$dr
  elif [ -z "${st[acc_t]:-}" ]; then st[acc_t]=$now; st[acc_a]=$acc; st[acc_d]=$dr; fi
else
  [ "$running" = true ] && alert "CRITICAL /metrics unreachable"
fi

# 3. dmesg Xid/NVRM and container errors (count-based, new since last run)
xid=$(sudo -n dmesg 2>/dev/null | grep -ciE "xid|nvrm")
[ -n "${st[xid]:-}" ] && [ "$xid" -gt "${st[xid]}" ] && alert "CRITICAL new Xid/NVRM lines in dmesg: $(sudo -n dmesg -T | grep -iE 'xid|nvrm' | tail -1 | cut -c1-160)"
st[xid]=$xid
if [ "$running" = true ]; then
  err=$(sudo -n docker logs --since 2m "$NAME" 2>&1 | grep -cE "ERROR|Traceback")
  [ "$err" -gt 0 ] && alert "WARN $err ERROR/Traceback lines in the last 2 min: $(sudo -n docker logs --since 2m "$NAME" 2>&1 | grep -E 'ERROR|Traceback' | tail -1 | cut -c1-160)"
fi

# 4. output
{ for k in "${!st[@]}"; do echo "$k=${st[$k]}"; done; } > "$STATE"
for a in "${alerts[@]}"; do
  echo "$ts $a" | tee -a "$ALERT_LOG"
  [ -n "$ALERT_CMD" ] && "$ALERT_CMD" "$ts $a" >/dev/null 2>&1
done
[ ${#alerts[@]} -eq 0 ] && [ "${VERBOSE:-0}" = 1 ] && echo "$ts OK health=$code waiting=${waiting:-?} running=${running_req:-?} preemptions=${pre:-?}"
exit 0
