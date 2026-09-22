#!/usr/bin/env python3
"""MTP acceptance per hour from the vLLM engine log (SpecDecoding metrics lines), plus request counts per client IP.
The /metrics counters are cumulative and dominated by periodic probes (e.g. a 1/min health check); this shows when real traffic happened.
Usage: docker logs glm53-nvfp4-prod 2>&1 | deploy/acceptance_by_hour.py [--k 3]   (k = num_speculative_tokens)
"""
import argparse, collections, re, sys

ap = argparse.ArgumentParser()
ap.add_argument("--k", type=int, default=3, help="draft tokens per step (MTP k)")
a = ap.parse_args()
rx = re.compile(r"INFO (\d\d-\d\d) (\d\d):\d\d:\d\d .*Accepted: (\d+) tokens, Drafted: (\d+) tokens")
rq = re.compile(r"INFO: +([0-9.]+):\d+ - \"POST /v1/(chat/)?completions")
hours = collections.defaultdict(lambda: [0, 0])
clients = collections.Counter()
for line in sys.stdin:
    m = rx.search(line)
    if m:
        h = f"{m.group(1)} {m.group(2)}h"
        hours[h][0] += int(m.group(3)); hours[h][1] += int(m.group(4))
        continue
    m = rq.search(line)
    if m:
        clients[m.group(1)] += 1
print("hour (UTC)   accepted   drafted   acceptance length (1 + k*accepted/drafted)")
ta = td = 0
for h in sorted(hours):
    acc, dr = hours[h]; ta += acc; td += dr
    if dr:
        print(f"{h}   {acc:8d}  {dr:8d}   {1 + a.k * acc / dr:.2f}")
if td:
    print(f"total         {ta:8d}  {td:8d}   {1 + a.k * ta / td:.2f}")
print("requests per client:", ", ".join(f"{ip}={n}" for ip, n in clients.most_common()))
