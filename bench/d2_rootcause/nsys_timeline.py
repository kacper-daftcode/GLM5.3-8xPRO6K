#!/usr/bin/env python3
"""Where on the session timeline do the profiler-signature kernels and the long blocked API calls lie?
Prints, relative to the first event of the trace (= container start): per-10-s histogram of `delayStreamKernel` launches per device,
the autotune-looking windows, and every CUDA API call blocked > 2 s. Use together with the container log (READY time, first/last
`Engine 000:` line) to tell warm-up autotuning from anything that happened while serving requests.
Usage: nsys_timeline.py nsys_hang.sqlite [--ready-s 242]
"""
import argparse, collections, sqlite3

ap = argparse.ArgumentParser(); ap.add_argument("db"); ap.add_argument("--ready-s", type=float, default=None, help="READY offset from trace start, s (from start.log)")
a = ap.parse_args()
cur = sqlite3.connect(a.db).cursor()
strings = dict(cur.execute("select id, value from StringIds"))
name = lambda k: strings.get(k, str(k))
K = "CUPTI_ACTIVITY_KIND_KERNEL"; R = "CUPTI_ACTIVITY_KIND_RUNTIME"
t0 = min(cur.execute(f"select min(start) from {K}").fetchone()[0], cur.execute(f"select min(start) from {R}").fetchone()[0])
t1 = max(cur.execute(f"select max(end) from {K}").fetchone()[0], cur.execute(f"select max(end) from {R}").fetchone()[0])
print(f"trace span {(t1 - t0) / 1e9:.1f} s" + (f"; READY at +{a.ready_s:.0f} s" if a.ready_s else ""))

sig = [k for k, v in strings.items() if v and ("delayStreamKernel" in v or "delay_kernel" in v.lower())]
if sig:
    rows = cur.execute(f"select deviceId, start, end from {K} where shortName in ({','.join('?' * len(sig))}) or demangledName in ({','.join('?' * len(sig))}) order by start", sig + sig).fetchall()
    print(f"\n{len(rows)} delayStreamKernel launches; first at +{(rows[0][1] - t0) / 1e9:.1f} s, last at +{(rows[-1][2] - t0) / 1e9:.1f} s" if rows else "\nno delayStreamKernel launches")
    hist = collections.Counter((r[0], int((r[1] - t0) / 1e10)) for r in rows)
    for (dev, b) in sorted(hist):
        print(f"  device {dev}: +{b * 10:4d}..{b * 10 + 10:4d} s  {hist[(dev, b)]:5d} launches")
else:
    print("\nno delayStreamKernel string in the trace")

print("\nCUDA API calls blocked > 2 s (thread-level):")
longs = cur.execute(f"select start, end, globalTid, nameId from {R} where end - start > 2000000000 order by start").fetchall()
for st, en, gtid, nid in longs:
    print(f"  +{(st - t0) / 1e9:7.1f} s  {(en - st) / 1e9:5.1f} s  pid={gtid >> 24} {name(nid)}")
print(f"{len(longs)} blocked calls; in [0, READY): {sum(1 for r in longs if a.ready_s and (r[0] - t0) / 1e9 < a.ready_s)}; after READY: {sum(1 for r in longs if a.ready_s and (r[0] - t0) / 1e9 >= a.ready_s)}")

print("\nkernel time per device per 30 s bucket (busy fraction):")
rows = cur.execute(f"select deviceId, start, end from {K}").fetchall()
busy = collections.defaultdict(float)
for dev, st, en in rows:
    b = int((st - t0) / 3e10); busy[(dev, b)] += (en - st) / 1e9
devs = sorted({d for d, _ in busy}); nb = int((t1 - t0) / 3e10) + 1
print("   bucket  " + " ".join(f"dev{d}" for d in devs))
for b in range(nb):
    print(f"  +{b * 30:4d} s  " + " ".join(f"{min(busy[(d, b)] / 30, 1.0):4.2f}" for d in devs))
