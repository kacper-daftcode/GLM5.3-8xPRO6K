#!/usr/bin/env python3
import sqlite3, sys
db = sqlite3.connect(sys.argv[1]); cur = db.cursor()
t0_rel, t1_rel = float(sys.argv[2]), float(sys.argv[3])
strings = dict(cur.execute("select id, value from StringIds")); name = lambda k: strings.get(k, str(k))
mx = cur.execute("select max(end) from CUPTI_ACTIVITY_KIND_RUNTIME").fetchone()[0]
t0, t1 = mx + int(t0_rel * 1e9), mx + int(t1_rel * 1e9)
for dev in [int(x) for x in sys.argv[4].split(",")]:
    rows = cur.execute("select shortName, count(*), sum(end-start)/1e6, min(gridX), max(gridX) from CUPTI_ACTIVITY_KIND_KERNEL where deviceId=? and start>=? and end<=? group by shortName order by 2 desc limit 14", (dev, t0, t1)).fetchall()
    tot = cur.execute("select count(*) from CUPTI_ACTIVITY_KIND_KERNEL where deviceId=? and start>=? and end<=?", (dev, t0, t1)).fetchone()[0]
    nccl = cur.execute("select shortName, streamId, count(*), sum(end-start)/1e6 from CUPTI_ACTIVITY_KIND_KERNEL where deviceId=? and start>=? and end<=? and shortName in (select id from StringIds where value like 'nccl%') group by shortName, streamId", (dev, t0, t1)).fetchall()
    graphs = cur.execute("select count(*) from CUPTI_ACTIVITY_KIND_KERNEL where deviceId=? and start>=? and end<=? and graphId is not null and graphId>0", (dev, t0, t1)).fetchone()[0] if any(r[1]=="graphId" for r in cur.execute("pragma table_info(CUPTI_ACTIVITY_KIND_KERNEL)")) else -1
    print(f"\n== device {dev}: {tot} kernels (in CUDA graphs: {graphs})")
    for r in rows:
        print(f"   {r[1]:6d} x {name(r[0])[:60]:60s} busy={r[2]:8.1f}ms grid[{r[3]},{r[4]}]")
    print("   nccl:", [(name(n)[:36], s, c, round(t, 1)) for n, s, c, t in nccl])
