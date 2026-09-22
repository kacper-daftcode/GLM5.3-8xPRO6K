#!/usr/bin/env python3
"""All devices/processes in a time window: idle gaps on stream 23, blocked API calls per thread (>1 s), event syncs."""
import sqlite3, sys, collections
db = sqlite3.connect(sys.argv[1]); cur = db.cursor()
t0_rel, t1_rel = float(sys.argv[2]), float(sys.argv[3])
strings = dict(cur.execute("select id, value from StringIds")); name = lambda k: strings.get(k, str(k))
mx = cur.execute("select max(end) from CUPTI_ACTIVITY_KIND_RUNTIME").fetchone()[0]
t0, t1 = mx + int(t0_rel * 1e9), mx + int(t1_rel * 1e9)
dev_of_pid = dict(cur.execute("select globalPid>>24, deviceId from CUPTI_ACTIVITY_KIND_KERNEL group by globalPid>>24"))
for pid, dev in sorted(dev_of_pid.items(), key=lambda x: x[1]):
    ops = cur.execute("select start,end from CUPTI_ACTIVITY_KIND_KERNEL where deviceId=? and streamId=23 and start>=? and end<=? order by start", (dev, t0, t1)).fetchall()
    gaps = sorted(((ops[i+1][0] - ops[i][1]) / 1e6, (ops[i][1]-mx)/1e9) for i in range(len(ops)-1))[-2:] if len(ops) > 1 else []
    longk = cur.execute("select shortName, streamId, (end-start)/1e9, (start-?)/1e9 from CUPTI_ACTIVITY_KIND_KERNEL where deviceId=? and start>=? and end<=? and end-start>1000000000 order by start", (mx, dev, t0, t1)).fetchall()
    api = cur.execute("select nameId, globalTid&0xffffff, (end-start)/1e9, (start-?)/1e9 from CUPTI_ACTIVITY_KIND_RUNTIME where globalTid>>24=? and end>=? and start<=? and end-start>1000000000 order by start", (mx, pid, t0, t1)).fetchall()
    print(f"\n== device {dev} pid {pid}: stream23 kernels={len(ops)} biggest idle gaps (ms,@t)={[(round(g,1), round(t,3)) for g,t in gaps]}")
    for k in longk:
        print(f"   long kernel {name(k[0])[:40]} stream={k[1]} dur={k[2]:.2f}s @t={k[3]:.2f}s")
    for a in api:
        print(f"   blocked API {name(a[0])} tid={a[1]} dur={a[2]:.2f}s @t={a[3]:.2f}s")
