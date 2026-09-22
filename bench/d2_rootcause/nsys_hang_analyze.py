#!/usr/bin/env python3
"""Analyze an nsys sqlite export of a hung run: per device, last GPU ops; per thread, API calls that never returned."""
import sqlite3, sys, collections

db = sqlite3.connect(sys.argv[1])
cur = db.cursor()
tables = {r[0] for r in cur.execute("select name from sqlite_master where type='table'")}
print("tables:", sorted(t for t in tables if t.startswith("CUPTI") or t in ("StringIds", "ProcessStreams", "TARGET_INFO_CUDA_STREAM"))[:20])
strings = dict(cur.execute("select id, value from StringIds"))
session_end = cur.execute("select max(end) from CUPTI_ACTIVITY_KIND_RUNTIME").fetchone()[0] if "CUPTI_ACTIVITY_KIND_RUNTIME" in tables else None
print("session end (ns):", session_end)

def name(k):
    return strings.get(k, str(k))

# ---- GPU ops: last ones per device ------------------------------------------------------------------------
ops = []
if "CUPTI_ACTIVITY_KIND_KERNEL" in tables:
    for r in cur.execute("select start,end,deviceId,streamId,shortName,gridX,gridY,gridZ,blockX from CUPTI_ACTIVITY_KIND_KERNEL"):
        ops.append((r[0], r[1], r[2], r[3], "K:" + name(r[4])[:70] + f" grid({r[5]},{r[6]},{r[7]}) blk{r[8]}"))
if "CUPTI_ACTIVITY_KIND_MEMCPY" in tables:
    for r in cur.execute("select start,end,deviceId,streamId,copyKind,bytes from CUPTI_ACTIVITY_KIND_MEMCPY"):
        ops.append((r[0], r[1], r[2], r[3], f"MEMCPY kind={r[4]} bytes={r[5]}"))
if "CUPTI_ACTIVITY_KIND_MEMSET" in tables:
    for r in cur.execute("select start,end,deviceId,streamId,bytes from CUPTI_ACTIVITY_KIND_MEMSET"):
        ops.append((r[0], r[1], r[2], r[3], f"MEMSET bytes={r[4]}"))
print(f"GPU ops total: {len(ops)}")
by_dev = collections.defaultdict(list)
for o in ops:
    by_dev[o[2]].append(o)
last_end_global = max(o[1] for o in ops)
print("\n=== per device: last GPU op end (ms before global last), last 6 ops ===")
for d in sorted(by_dev):
    L = sorted(by_dev[d], key=lambda o: o[1])
    last = L[-1]
    print(f"\n-- device {d}: {len(L)} ops, last end = {(last_end_global - last[1]) / 1e6:9.1f} ms before global last op")
    for o in L[-6:]:
        print(f"   start={(o[0]-last_end_global)/1e6:10.1f}ms dur={(o[1]-o[0])/1e3:9.1f}us stream={o[3]:4d} {o[4]}")
    # ops with long duration (>= 1s) = spinning kernels
    longs = [o for o in L if o[1] - o[0] > 1e9]
    for o in longs[-3:]:
        print(f"   LONG: start={(o[0]-last_end_global)/1e6:10.1f}ms dur={(o[1]-o[0])/1e9:6.2f}s stream={o[3]} {o[4]}")

# ---- API calls never returned (end >= session end - small) -------------------------------------------------
print("\n=== CUDA API calls still in flight at the end (per thread) ===")
for tbl in ("CUPTI_ACTIVITY_KIND_RUNTIME", "CUPTI_ACTIVITY_KIND_DRIVER"):
    if tbl not in tables:
        continue
    mx = cur.execute(f"select max(end) from {tbl}").fetchone()[0]
    rows = cur.execute(f"select start,end,globalTid,nameId,correlationId from {tbl} where end >= ? order by start", (mx - 5_000_000,)).fetchall()
    print(f"-- {tbl}: max end {mx}, {len(rows)} calls ending in the last 5 ms")
    seen = set()
    for r in rows:
        tid = r[2]
        if tid in seen:
            continue
        seen.add(tid)
        print(f"   tid={tid} {name(r[3])} start={(r[0]-mx)/1e6:10.1f}ms dur={(r[1]-r[0])/1e6:9.1f}ms corr={r[4]}")
    # longest API calls overall (blocked launches show up as very long)
    rows = cur.execute(f"select start,end,globalTid,nameId from {tbl} order by (end-start) desc limit 12").fetchall()
    print(f"-- {tbl}: longest calls")
    for r in rows:
        print(f"   tid={r[2]} {name(r[3])} dur={(r[1]-r[0])/1e6:9.1f}ms start={(r[0]-mx)/1e6:10.1f}ms")

# ---- per stream: last op and whether a kernel launch exists without a GPU record ------------------------------
print("\n=== per device: GPU idle gap at the end (time from last op end to session end) ===")
for d in sorted(by_dev):
    L = sorted(by_dev[d], key=lambda o: o[1])
    print(f"   device {d}: last op ended {(session_end - L[-1][1])/1e6 if session_end else float('nan'):9.1f} ms before session end; last op: {L[-1][4][:60]}")
