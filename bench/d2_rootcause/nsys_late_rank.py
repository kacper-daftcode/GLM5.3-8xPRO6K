#!/usr/bin/env python3
"""Detail of the late rank during a stall window: kernels/memcpys by stream, memcpy kinds/sizes/durations."""
import sqlite3, sys, collections
db = sqlite3.connect(sys.argv[1]); cur = db.cursor()
strings = dict(cur.execute("select id, value from StringIds")); name = lambda k: strings.get(k, str(k))
mx = cur.execute("select max(end) from CUPTI_ACTIVITY_KIND_RUNTIME").fetchone()[0]
KIND = {1: "H2D", 2: "D2H", 8: "D2D", 10: "P2P", 0: "?"}
for dev, t0_rel, t1_rel in [(4, -30.3, -21.5), (5, -15.5, -6.8), (0, -212.0, -198.5)]:
    t0, t1 = mx + int(t0_rel * 1e9), mx + int(t1_rel * 1e9)
    print(f"\n===== device {dev}, window [{t0_rel}s, {t1_rel}s] =====")
    ks = cur.execute("select streamId, count(*), sum(end-start)/1e6 from CUPTI_ACTIVITY_KIND_KERNEL where deviceId=? and start>=? and end<=? group by streamId", (dev, t0, t1)).fetchall()
    print("kernels by stream (count, busy ms):", ks)
    top = cur.execute("select shortName, count(*), sum(end-start)/1e6 from CUPTI_ACTIVITY_KIND_KERNEL where deviceId=? and start>=? and end<=? group by shortName order by 3 desc limit 8", (dev, t0, t1)).fetchall()
    print("top kernels by busy time:", [(name(n)[:38], c, round(t, 1)) for n, c, t in top])
    ms = cur.execute("select copyKind, streamId, count(*), sum(bytes)/1048576.0, sum(end-start)/1e6, max(end-start)/1e6 from CUPTI_ACTIVITY_KIND_MEMCPY where deviceId=? and start>=? and end<=? group by copyKind, streamId", (dev, t0, t1)).fetchall()
    print("memcpys (kind, stream, count, MiB, total ms, max ms):", [(KIND.get(k, k), s, c, round(mb, 1), round(t, 1), round(mxd, 1)) for k, s, c, mb, t, mxd in ms])
    # timeline of memcpys > 1 MiB
    rows = cur.execute("select start,end,copyKind,bytes,streamId from CUPTI_ACTIVITY_KIND_MEMCPY where deviceId=? and start>=? and end<=? and bytes>1048576 order by start limit 40", (dev, t0, t1)).fetchall()
    for r in rows[:12]:
        print(f"   memcpy {KIND.get(r[2], r[2])} {r[3]/1048576:7.1f} MiB start={(r[0]-mx)/1e9:9.3f}s dur={(r[1]-r[0])/1e6:8.2f}ms stream={r[4]}")
    # gaps: biggest idle gaps between consecutive GPU ops on the compute stream 23 in the window
    ops = cur.execute("select start,end from CUPTI_ACTIVITY_KIND_KERNEL where deviceId=? and streamId=23 and start>=? and end<=? order by start", (dev, t0, t1)).fetchall()
    gaps = sorted(((ops[i+1][0] - ops[i][1]) / 1e6, (ops[i][1]-mx)/1e9) for i in range(len(ops)-1))[-4:]
    print("largest idle gaps on stream 23 (ms, at t):", [(round(g, 1), round(t, 3)) for g, t in gaps])
    # which NCCL kernels ran on this device in the window
    nc = cur.execute("select shortName, streamId, count(*), sum(end-start)/1e6, max(end-start)/1e6 from CUPTI_ACTIVITY_KIND_KERNEL where deviceId=? and start>=? and end<=? and shortName in (select id from StringIds where value like 'nccl%') group by shortName, streamId", (dev, t0, t1)).fetchall()
    print("nccl kernels:", [(name(n)[:34], s, c, round(t, 1), round(m, 1)) for n, s, c, t, m in nc])
