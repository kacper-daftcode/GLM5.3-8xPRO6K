#!/usr/bin/env python3
"""Podsumowanie JSONL soaku: soak_status.py results/soak_seg*.jsonl"""
import collections
import json
import sys

for f in sys.argv[1:]:
    rows = [json.loads(l) for l in open(f) if l.strip()]
    if not rows:
        print(f, "empty")
        continue
    kinds = collections.Counter(r["kind"] for r in rows)
    mon = [r for r in rows if r["kind"] == "monitor"]
    longs = [r for r in rows if r["kind"] == "long"]
    errs = [r for r in rows if r["kind"] in ("error", "ALERT")]
    warns = [r for r in rows if r["kind"] == "warn"]
    bursts = [r for r in rows if r["kind"] == "burst"]
    summ = [r for r in rows if r["kind"] == "summary"]
    acc = [m["acc"] for m in mon if m.get("acc")]
    gen = [m["gen_tps"] for m in mon if m.get("gen_tps") is not None]
    wait = [m.get("waiting", 0) for m in mon]
    run = [m.get("running", 0) for m in mon]
    kv = [m.get("kv_pct", 0) for m in mon]
    tmax = [m["temp_max"] for m in mon if m.get("temp_max")]
    rss = [m["rss"] for m in mon if m.get("rss")]
    print(f"== {f}: {rows[0]['ts']} -> {rows[-1]['ts']} lines={len(rows)} kinds={dict(kinds)}")
    if mon:
        acc_s = f"{min(acc):.2f}/{sum(acc) / len(acc):.2f}/{max(acc):.2f}" if acc else "-"
        gen_s = f"{sum(gen) / len(gen):.0f} max {max(gen):.0f}" if gen else "-"
        rss_s = f"{rss[0]} -> {rss[-1]}" if rss else ""
        print(f"   monitor: running avg {sum(run) / len(run):.1f} max {max(run):.0f} | waiting max {max(wait):.0f} | KV% max {max(kv):.1f} | "
              f"gen tok/s avg {gen_s} | acc min/avg/max {acc_s} | temp max {max(tmax) if tmax else 0} | rss {rss_s} | "
              f"dmesg_new {mon[-1].get('dmesg_new')} log_err_new {mon[-1].get('log_err_new')}")
    if longs:
        ok = sum(1 for r in longs if r["needle_ok"])
        tps = [r["toks_per_s"] for r in longs]
        print(f"   long: {len(longs)} (needle ok {ok}) tok/s min/avg {min(tps)}/{sum(tps) // len(tps)}; lens {sorted(set(round(r['tokens'] / 1000) for r in longs))}k")
        for r in longs:
            if not r["needle_ok"]:
                print(f"     FAIL {r['ts']} {r['tokens']} tok: {r['text'][:300]!r}")
    if bursts:
        print(f"   bursts: {len(bursts)} ok {sum(b['ok'] for b in bursts)}/{sum(b['size'] for b in bursts)} secs max {max(b['secs'] for b in bursts)} p95 max {max(b['p95'] for b in bursts)}")
    print(f"   errors/alerts: {len(errs)} warns: {len(warns)}")
    for r in errs[-5:]:
        print("     ", json.dumps(r, ensure_ascii=False)[:300])
    for r in warns[-3:]:
        print("     warn", json.dumps(r, ensure_ascii=False)[:200])
    for r in summ:
        print("   summary:", json.dumps(r["stats"]), "alive", r["alive_threads"])
        print("   lat:", json.dumps(r["lat"]))
