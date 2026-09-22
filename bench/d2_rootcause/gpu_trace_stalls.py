#!/usr/bin/env python3
"""Stall / hang classifier for the 1 Hz GPU trace written by exp.sh (nvidia-smi -lms 1000):
   timestamp, index, utilization.gpu [%], power.draw [W], memory.used [MiB]

Per sampling round each PRO GPU is classified from hardware counters only (the engine log is silent during long prefills):
  busy  : power >= BUSY_W  (a live prefill pins the card at the 300 W cap)
  wait  : util >= 90 % and power < WAIT_W  (NCCL spin-wait: 100 % "utilization" at ~130 W)
  idle  : util < 50 % and power < IDLE_W   (nothing resident)
  other : everything else (decode is memory-bound: high util at intermediate power)
Round classes: prefill (all busy) | desync (>= 1 busy and >= 1 wait/idle *within the same TP group*; the E32 nsys signature: one rank
computing while its three peers wait) | spin (no busy, >= 1 wait) | idle | other. Windows of consecutive desync/spin rounds >= MIN_WIN s
are listed with timestamps so they can be matched against the reproducer log (decode phases are short and print status lines).
Usage: gpu_trace_stalls.py gpu_trace.csv [--gpus 0,1,2,3,4,5,7,8] [--tp 4]
"""
import argparse, collections, sys

ap = argparse.ArgumentParser()
ap.add_argument("csv"); ap.add_argument("--gpus", default="0,1,2,3,4,5,7,8", help="PRO GPUs in rank order (stage 0 first)")
ap.add_argument("--tp", type=int, default=4); ap.add_argument("--busy-w", type=float, default=250); ap.add_argument("--wait-w", type=float, default=180)
ap.add_argument("--idle-w", type=float, default=100); ap.add_argument("--min-win", type=int, default=3)
a = ap.parse_args()
gpus = [int(x) for x in a.gpus.split(",")]
group_of = {g: i // a.tp for i, g in enumerate(gpus)}

rounds = []  # (timestamp, {gpu: (util, power)})
cur = {}; cur_ts = None
for line in open(a.csv):
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 4 or not parts[1].isdigit():
        continue
    try:
        idx, util, power = int(parts[1]), float(parts[2]), float(parts[3])
    except ValueError:
        continue
    if idx not in group_of:
        continue
    if idx in cur:  # index repeats -> new round
        rounds.append((cur_ts, cur)); cur = {}
    if not cur:
        cur_ts = parts[0]
    cur[idx] = (util, power)
if cur:
    rounds.append((cur_ts, cur))


def cls(util, power):
    if power >= a.busy_w: return "busy"
    if util >= 90 and power < a.wait_w: return "wait"
    if util < 50 and power < a.idle_w: return "idle"
    return "other"


def round_class(sample):
    c = {g: cls(*sample[g]) for g in sample}
    if not c: return "empty", c
    vals = set(c.values())
    if vals == {"busy"}: return "prefill", c
    if vals == {"idle"}: return "idle", c
    if "busy" in vals:
        for grp in set(group_of.values()):
            members = [g for g in c if group_of[g] == grp]
            gc = {c[g] for g in members}
            if "busy" in gc and gc & {"wait", "idle"}:
                return "desync", c
        return "mixed", c
    if "wait" in vals: return "spin", c
    return "other", c


counts = collections.Counter(); windows = []; run = None
for ts, sample in rounds:
    rc, c = round_class(sample)
    counts[rc] += 1
    if rc in ("desync", "spin"):
        if run and run["class"] == rc:
            run["n"] += 1; run["busy"] |= {g for g in c if c[g] == "busy"}; run["wait"] |= {g for g in c if c[g] in ("wait", "idle")}
        else:
            if run and run["n"] >= a.min_win: windows.append(run)
            run = {"class": rc, "start": ts, "n": 1, "busy": {g for g in c if c[g] == "busy"}, "wait": {g for g in c if c[g] in ("wait", "idle")}}
    else:
        if run and run["n"] >= a.min_win: windows.append(run)
        run = None
if run and run["n"] >= a.min_win: windows.append(run)

print(f"rounds={len(rounds)} (1 Hz)  classes: " + " ".join(f"{k}={v}s" for k, v in sorted(counts.items(), key=lambda kv: -kv[1])))
for w in windows:
    print(f"  {w['class']:6s} {w['start']} {w['n']:4d}s  busy={sorted(w['busy'])} waiting/idle={sorted(w['wait'])}")
des = [w for w in windows if w["class"] == "desync"]; spn = [w for w in windows if w["class"] == "spin"]
print(f"summary: prefill={counts['prefill']}s desync_windows={len(des)} (>= {a.min_win}s; max {max([w['n'] for w in des], default=0)}s, total {sum(w['n'] for w in des)}s) "
      f"spin_windows={len(spn)} (total {sum(w['n'] for w in spn)}s) idle={counts['idle']}s")
