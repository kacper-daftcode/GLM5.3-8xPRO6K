#!/usr/bin/env python3
"""For each long blocked CUDA API call (>2 s), list what every thread of the same process did in that window."""
import sqlite3, sys, collections

db = sqlite3.connect(sys.argv[1]); cur = db.cursor()
strings = dict(cur.execute("select id, value from StringIds"))
name = lambda k: strings.get(k, str(k))
T = "CUPTI_ACTIVITY_KIND_RUNTIME"
mx = cur.execute(f"select max(end) from {T}").fetchone()[0]
longs = cur.execute(f"select start,end,globalTid,nameId from {T} where end-start > 2000000000 order by start").fetchall()
print(f"{len(longs)} API calls longer than 2 s")
pid_of = lambda gtid: gtid >> 24
for st, en, gtid, nid in longs:
    pid = pid_of(gtid)
    print(f"\n=== {name(nid)} tid={gtid & 0xffffff} pid={pid} dur={(en-st)/1e9:.2f}s at t={(st-mx)/1e9:.1f}s")
    # other API calls in this process overlapping [st-0.5s, en]
    rows = cur.execute(f"select start,end,globalTid,nameId from {T} where globalTid>>24 = ? and end >= ? and start <= ? and globalTid != ? order by start", (pid, st - 500_000_000, en, gtid)).fetchall()
    by_thr = collections.defaultdict(list)
    for r in rows:
        by_thr[r[2] & 0xffffff].append(r)
    for thr, rs in by_thr.items():
        big = [r for r in rs if r[1] - r[0] > 50_000_000]  # >50 ms
        cnt = collections.Counter(name(r[3]) for r in rs)
        print(f"  thread {thr}: {len(rs)} calls; top: {cnt.most_common(4)}")
        for r in big[:6]:
            print(f"     LONG {name(r[3])} dur={(r[1]-r[0])/1e6:.0f}ms start={(r[0]-st)/1e6:+.0f}ms rel. to blocked call")
    # what did THIS thread do right before the blocked call
    prev = cur.execute(f"select start,end,nameId from {T} where globalTid = ? and end <= ? order by end desc limit 8", (gtid, st)).fetchall()
    print("  same thread, calls right before:", [f"{name(p[2])}({(p[1]-p[0])/1e6:.1f}ms)" for p in reversed(prev)])
    # GPU ops on this process's device during the window: any memcpy/memset/kernel running?
    dev = cur.execute("select deviceId from CUPTI_ACTIVITY_KIND_KERNEL where globalPid>>24 = ? limit 1", (pid,)).fetchone()
    if dev:
        d = dev[0]
        k = cur.execute("select count(*), min(start), max(end) from CUPTI_ACTIVITY_KIND_KERNEL where deviceId=? and start>=? and end<=?", (d, st, en)).fetchone()
        m = cur.execute("select count(*), sum(bytes) from CUPTI_ACTIVITY_KIND_MEMCPY where deviceId=? and start>=? and end<=?", (d, st, en)).fetchone()
        running = cur.execute("select shortName,start,end,streamId from CUPTI_ACTIVITY_KIND_KERNEL where deviceId=? and start<? and end>? order by end-start desc limit 3", (d, st + 1_000_000_000, st + 1_000_000_000)).fetchall()
        print(f"  device {d} during window: kernels={k[0]} memcpys={m[0]} ({(m[1] or 0)/2**20:.0f} MiB); kernels spanning t+1s: {[(name(r[0])[:40], (r[2]-r[1])/1e9, r[3]) for r in running]}")
